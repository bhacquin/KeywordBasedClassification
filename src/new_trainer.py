import warnings
warnings.filterwarnings("ignore")
import pickle as p
from collections import defaultdict
import time
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from math import ceil
import torch
from torch import nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, WeightedRandomSampler
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
from nltk.corpus import stopwords

import numpy as np
import os
import shutil
import sys
from tqdm import tqdm
from model import LOTClassModel
import pandas as pd
import numpy as np
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_bert import BertOnlyMLMHead    
import random
from nltk.corpus import wordnet as wn
## TO DO : IMPLEMENT THE CASE WHEN NO GROUND TRUTH IS AVAILABLE
## TO DO : MULTI GPU
## TO DO : MASK KEYWORD

## TO DO : DO NOT ADD EXTRA CLASS IF NOT ENOUGH DATA POINTS

## UTILITARY FUNCTIONS

def is_adj(word):
    for ss in wn.synsets(word):
        if wn.ADJ not in ss.pos():
            continue
        else:
            return True
    return False


def antonyms_for(word):
    antonyms = set()
    for ss in wn.synsets(word):
        
        for lemma in ss.lemmas():
            any_pos_antonyms = [ antonym.name() for antonym in lemma.antonyms() ]
        for antonym in any_pos_antonyms:
                antonym_synsets = wn.synsets(antonym)
                if wn.ADJ not in [ ss.pos() for ss in antonym_synsets ]:
                    continue
                antonyms.add(antonym)
    return list(antonyms)


## DEFINITION OF THE MODEL 
class ClassifModel(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self.classifier = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                        nn.Tanh(),
                                        nn.Dropout(config.hidden_dropout_prob),
                                        nn.Linear(config.hidden_size, config.num_labels))
        self.init_weights()
        # MLM head is not trained
        for param in self.cls.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids, pred_mode, attention_mask=None, token_type_ids=None, 
                position_ids=None, head_mask=None, inputs_embeds=None):
        bert_outputs = self.bert(input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids,
                                head_mask=head_mask,
                                inputs_embeds=inputs_embeds)
        last_hidden_states = bert_outputs[0]
        if pred_mode == "classification":
            logits = self.classifier(last_hidden_states)
        elif pred_mode == "mlm":
            logits = self.cls(last_hidden_states)
        else:
            sys.exit("Wrong pred_mode!")
        return logits


class ClassifTrainer(object):

    def __init__(self, args):
        self.args = args
        self.max_len = args.max_len
        self.dataset_dir = args.dataset_dir
        self.dist_port = args.dist_port
        self.num_cpus = min(10, cpu_count() - 1) if cpu_count() > 1 else 1
        self.world_size = args.gpus
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.accum_steps = args.accum_steps
        eff_batch_size = self.train_batch_size * self.world_size * self.accum_steps
        assert abs(eff_batch_size - 128) < 10, f"Make sure the effective training batch size is around 128, current: {eff_batch_size}"
        print(f"Effective training batch size: {eff_batch_size}")
        self.pretrained_lm = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_lm, do_lower_case=True)
        self.vocab = self.tokenizer.get_vocab()
        self.vocab_size = len(self.vocab)
        self.mask_id = self.vocab[self.tokenizer.mask_token]
        self.inv_vocab = {k:v for v, k in self.vocab.items()}
        self.read_label_names(args.dataset_dir, args.label_names_file, check_antonym=True)
        self.num_class = len(self.label_name_dict) + 1 ### Class pointing to each keyword and 1 for the rest assumed negative
        self.num_keywords = len(self.label_name_dict)
        self.minimum_occurences_per_class = 1000
        self.occurences_per_class = self.count_occurences(args.dataset_dir, args.train_file)

        self.model = ClassifModel.from_pretrained(self.pretrained_lm,
                                                   output_attentions=False,
                                                   output_hidden_states=False,
                                                   num_labels=self.num_class) 
        self.with_train_label = True if args.train_label_file is not None else False
        self.train_text_file = args.train_file
        self.read_data(args.dataset_dir, args.train_file,args.train_label_file, args.test_file, args.test_label_file)
        self.with_test_label = True if args.test_label_file is not None else False ### bizarre
        self.temp_dir = f'tmp_{self.dist_port}'
        self.mcp_loss = nn.CrossEntropyLoss()
        self.st_loss = nn.KLDivLoss(reduction='batchmean')
        self.update_interval = args.update_interval
        self.early_stop = args.early_stop   ### TO DO : check minimum size
        self.non_accepted_words = []
        self.vocab_loop_counter = 0 ### TO DO : move to the right method
        
        
        self.loop_over_vocab = args.loop_over_vocab  ### 
        self.look_for_negative = True
        
        self.label_names_used = {}

        self.true_label = [int(i) for i in str(args.true_label).split(' ')]
        print('True Label', self.true_label)
        print('occurences', self.occurences_per_class)
        self.verbose = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        #keywords ### TO DO CHECK IF NECESSARY? REPLACE BY self.true_label
        self.positive_keywords = [2]
        self.negative_keywords = [] ### you give a keyword that is assign to a negative label
        
        

    def add_positive_keyword(self, keyword):
        self.positive_keywords.append(keyword)

    def add_negative_keyword(self, keyword):
        self.negative_keywords.append(keyword)


    # set up distributed training
    def set_up_dist(self, rank):
        dist.init_process_group(
            backend='nccl',
            init_method=f'tcp://localhost:{self.dist_port}',
            world_size=self.world_size,
            rank=rank
        )
        # create local model
        model = self.model.to(rank)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        return model

    # get document truncation statistics with the defined max length
    def corpus_trunc_stats(self, docs):
        doc_len = []
        for doc in docs:
            input_ids = self.tokenizer.encode(doc, add_special_tokens=True)
            doc_len.append(len(input_ids))
        print(f"Document max length: {np.max(doc_len)}, avg length: {np.mean(doc_len)}, std length: {np.std(doc_len)}")
        trunc_frac = np.sum(np.array(doc_len) > self.max_len) / len(doc_len)
        print(f"Truncated fraction of all documents: {trunc_frac}")

    # Count occurences of keywords in corpus
    def count_occurences(self,dataset_dir, text_file, dict_of_keywords = None):
        
        if dict_of_keywords is None:
            assert(self.label_name_dict is not None)
            dict_of_keywords = self.label_name_dict
        corpus = open(os.path.join(dataset_dir, text_file), encoding="utf-8")
        docs = [doc.strip() for doc in corpus.readlines()]
        occurences_per_class = {i:0 for i in dict_of_keywords}
        for doc in docs:
            for i in dict_of_keywords:
                for keyword in self.label_name_dict[i]:
                    occurences_per_class[i] += int(keyword.lower() in doc.lower()[:self.max_len])
        print('occurences_per_class',occurences_per_class)
        return occurences_per_class


    # convert a list of strings to token ids
    def encode(self, docs):
        encoded_dict = self.tokenizer.batch_encode_plus(docs, add_special_tokens=True, max_length=self.max_len, padding='max_length',
                                                        return_attention_mask=True, truncation=True, return_tensors='pt')
        input_ids = encoded_dict['input_ids']
        attention_masks = encoded_dict['attention_mask']
        return input_ids, attention_masks

    def non_batch_encode(self, docs):
        encoded_dict = self.tokenizer.encode_plus(docs, add_special_tokens=True, max_length=self.max_len, padding='max_length',
                                                        return_attention_mask=True, truncation=True, return_tensors='pt')
        input_ids = encoded_dict['input_ids']
        attention_masks = encoded_dict['attention_mask']
        return input_ids, attention_masks

    # convert list of token ids to list of strings
    def decode(self, ids):
        strings = self.tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return strings

    # convert dataset into tensors
    def create_dataset(self, dataset_dir, text_file, label_file, loader_name, find_label_name=False, label_name_loader_name=None, check_exist=True):
        print('GPU AVAILABLE : ', torch.cuda.is_available())
        loader_file = os.path.join(dataset_dir, loader_name)
        if os.path.exists(loader_file) and check_exist:
            print(f"Loading encoded texts from {loader_file}")
            data = torch.load(loader_file)
            print('Defining self.train_docs')
            corpus = open(os.path.join(dataset_dir, text_file), encoding="utf-8")
            docs = [doc.strip() for doc in corpus.readlines()]
            self.train_docs = docs
            if label_file is not None:
                print(f"Reading labels from {os.path.join(dataset_dir, label_file)}")
                truth = open(os.path.join(dataset_dir, label_file))
                labels = [int(label.strip()) for label in truth.readlines()]
                self.docs_labels = labels
        else:
            print(f"Reading texts from {os.path.join(dataset_dir, text_file)}")
            corpus = open(os.path.join(dataset_dir, text_file), encoding="utf-8")
            docs = [doc.strip() for doc in corpus.readlines()]
            self.train_docs = docs
            print(f"Converting texts into tensors.")
            chunk_size = ceil(len(docs) / self.num_cpus)
            chunks = [docs[x:x+chunk_size] for x in range(0, len(docs), chunk_size)]
            results = Parallel(n_jobs=self.num_cpus)(delayed(self.encode)(docs=chunk) for chunk in chunks)
            input_ids = torch.cat([result[0] for result in results])
            attention_masks = torch.cat([result[1] for result in results])
            print(f"Saving encoded texts into {loader_file}")
            if label_file is not None:
                print(f"Reading labels from {os.path.join(dataset_dir, label_file)}")
                truth = open(os.path.join(dataset_dir, label_file))
                labels = [int(label.strip()) for label in truth.readlines()]
                self.docs_labels = labels
                labels = torch.tensor(labels)
                data = {"input_ids": input_ids, "attention_masks": attention_masks, "labels": labels}
            else:
                data = {"input_ids": input_ids, "attention_masks": attention_masks}
            torch.save(data, loader_file)
        if find_label_name:
            loader_file = os.path.join(dataset_dir, label_name_loader_name)
            if os.path.exists(loader_file) and check_exist:
                print(f"Loading texts with label names from {loader_file}")
                label_name_data = torch.load(loader_file)
            else:
                print(f"Reading texts from {os.path.join(dataset_dir, text_file)}")
                corpus = open(os.path.join(dataset_dir, text_file), encoding="utf-8")
                docs = [doc.strip() for doc in corpus.readlines()]

                print("Locating label names in the corpus.")
                chunk_size = ceil(len(docs) / self.num_cpus)
                chunks = [docs[x:x+chunk_size] for x in range(0, len(docs), chunk_size)]
                results = Parallel(n_jobs=self.num_cpus)(delayed(self.label_name_occurrence)(docs=chunk) for chunk in chunks)
                input_ids_with_label_name = torch.cat([result[0] for result in results])
                attention_masks_with_label_name = torch.cat([result[1] for result in results])
                label_name_idx = torch.cat([result[2] for result in results])
                assert len(input_ids_with_label_name) > 0, "No label names appear in corpus!"
                label_name_data = {"input_ids": input_ids_with_label_name, "attention_masks": attention_masks_with_label_name, "labels": label_name_idx}
                loader_file = os.path.join(dataset_dir, label_name_loader_name)
                print(f"Saving texts with label names into {loader_file}")
                torch.save(label_name_data, loader_file)
            return data, label_name_data
        else:
            return data
    
    # find label name indices and replace out-of-vocab label names with [MASK]
    def label_name_in_doc(self, doc):
        doc = self.tokenizer.tokenize(doc)
        label_idx = -1 * torch.ones(self.max_len, dtype=torch.long)
        new_doc = []
        wordpcs = []
        idx = 1 # index starts at 1 due to [CLS] token
        for i, wordpc in enumerate(doc):
            wordpcs.append(wordpc[2:] if wordpc.startswith("##") else wordpc)
            if idx >= self.max_len - 1: # last index will be [SEP] token
                break
            if i == len(doc) - 1 or not doc[i+1].startswith("##"):
                word = ''.join(wordpcs)
                if word in self.label2class:
                    label_idx[idx] = self.label2class[word]                    
                    
                    # replace label names that are not in tokenizer's vocabulary with the [MASK] token
                    if word not in self.vocab:
                        wordpcs = [self.tokenizer.mask_token]
                new_word = ''.join(wordpcs)
                if new_word != self.tokenizer.unk_token:
                    idx += len(wordpcs)
                    new_doc.append(new_word)
                wordpcs = []
        if (label_idx >= 0).any():
            return ' '.join(new_doc), label_idx
        else:
            return None

    # find label name occurrences in the corpus
    def label_name_occurrence(self, docs):
        text_with_label = []
        label_name_idx = []
        for doc in docs:
            result = self.label_name_in_doc(doc)
            if result is not None:
                text_with_label.append(result[0])
                label_name_idx.append(result[1].unsqueeze(0))                    
        if len(text_with_label) > 0:
            encoded_dict = self.tokenizer.batch_encode_plus(text_with_label, add_special_tokens=True, max_length=self.max_len, 
                                                            padding='max_length', return_attention_mask=True, truncation=True, return_tensors='pt')
            input_ids_with_label_name = encoded_dict['input_ids']
            attention_masks_with_label_name = encoded_dict['attention_mask']
            label_name_idx = torch.cat(label_name_idx, dim=0)
        else:
            input_ids_with_label_name = torch.ones(0, self.max_len, dtype=torch.long)
            attention_masks_with_label_name = torch.ones(0, self.max_len, dtype=torch.long)
            label_name_idx = torch.ones(0, self.max_len, dtype=torch.long)
        return input_ids_with_label_name, attention_masks_with_label_name, label_name_idx

    # read text corpus and labels from files
    def read_data(self, dataset_dir, train_file, train_label_file, test_file, test_label_file, check_exist = True):
        self.train_data, self.label_name_data = self.create_dataset(dataset_dir, train_file, train_label_file, "train.pt", 
                                                                    find_label_name=True, label_name_loader_name="label_name_data.pt", check_exist = check_exist)
        if test_file is not None:
            self.test_data = self.create_dataset(dataset_dir, test_file, test_label_file, "test.pt", check_exist = check_exist)

    # read label names from file
    def read_label_names(self, dataset_dir, label_name_file, change_positivity_var = True, check_antonym = False):
        label_name_file = open(os.path.join(dataset_dir, label_name_file))
        labels_names_and_class = [label_name.split(';') for label_name in label_name_file.readlines()]
        label_names = [name[0].strip().split(',') for name in labels_names_and_class]
        positive_or_negative = [name[1] for name in labels_names_and_class]
        all_names = []
        for i in label_names:
            all_names += i
        ### Add antonyms if adj only:
        if check_antonym:
            for i,category_words in enumerate(label_names):
                for keyword in category_words:
                    opposite_already_here = False
                    if is_adj(keyword):
                        antonyms_keyword = antonyms_for(keyword)
                        for antonym in antonyms_keyword:
                            if antonym in all_names:
                                opposite_already_here = True
                                break
                        if opposite_already_here:
                            break
                   
                        label_names.append(antonyms_keyword)
                        if positive_or_negative[i].replace('\n','') == 'positive':
                            positive_or_negative.append('negative')

                        else:
                            positive_or_negative.append('positive')
                        break

                    


        self.label_name_dict = {i: [word.lower() for word in category_words] for i, category_words in enumerate(label_names)}
        label_name_positivity = {i: int((positive.replace('\n','') == 'positive')) for i, positive in enumerate(positive_or_negative)}
        label_name_positivity[len(positive_or_negative)] = 0
        if change_positivity_var:
            self.label_name_positivity = label_name_positivity
        print(f"Label names used for each class are: {self.label_name_dict}")
        print((f'Label names positivity is: {self.label_name_positivity}'))
        self.label2class = {}
        self.all_label_name_ids = [self.mask_id]
        self.all_label_names = [self.tokenizer.mask_token]
        for class_idx in self.label_name_dict:
            for word in self.label_name_dict[class_idx]:
                assert word not in self.label2class, f"\"{word}\" used as the label name by multiple classes!"
                self.label2class[word] = class_idx
                if word in self.vocab:
                    self.all_label_name_ids.append(self.vocab[word])
                    self.all_label_names.append(word)

    # create dataset loader
    def make_dataloader(self, rank, data_dict, batch_size):
        if "labels" in data_dict:
            dataset = TensorDataset(data_dict["input_ids"], data_dict["attention_masks"], data_dict["labels"])
        else:
            dataset = TensorDataset(data_dict["input_ids"], data_dict["attention_masks"])
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=rank)
        dataset_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, shuffle=False)
        return dataset_loader

    # filter out stop words and words in multiple categories
    def filter_keywords(self, category_vocab_size=100):
        all_words = defaultdict(list)
        sorted_dicts = {}
        for i, cat_dict in self.category_words_freq.items():
            sorted_dict = {k:v for k, v in sorted(cat_dict.items(), key=lambda item: item[1], reverse=True)[:category_vocab_size]}
            sorted_dicts[i] = sorted_dict
            for word_id in sorted_dict:
                all_words[word_id].append(i)
        repeat_words = []
        for word_id in all_words:
            if len(all_words[word_id]) > 1:
                repeat_words.append(word_id)
        self.category_vocab = {}
        for i, sorted_dict in sorted_dicts.items():
            self.category_vocab[i] = np.array(list(sorted_dict.keys()))
        stopwords_vocab = stopwords.words('english')
        for i, word_list in self.category_vocab.items():
            delete_idx = []
            for j, word_id in enumerate(word_list):
                word = self.inv_vocab[word_id]
                if word in self.label_name_dict[i]:
                    continue
                if not word.isalpha() or len(word) == 1 or word in stopwords_vocab or word_id in repeat_words:
                    delete_idx.append(j)
            self.category_vocab[i] = np.delete(self.category_vocab[i], delete_idx)

    # construct category vocabulary (distributed function)
    def category_vocabulary_dist(self, rank , top_pred_num=50, loader_name="category_vocab.pt"):
        model = self.set_up_dist(rank)
        model.eval()
        label_name_dataset_loader = self.make_dataloader(rank, self.label_name_data, self.eval_batch_size)
        category_words_freq = {i: defaultdict(float) for i in range(self.num_keywords)} ### the -1 comes from the addition of extra "negative" class based on no keyword
        wrap_label_name_dataset_loader = tqdm(label_name_dataset_loader) if rank == 0 else label_name_dataset_loader
        try:
            for batch in wrap_label_name_dataset_loader:
                with torch.no_grad():
                    input_ids = batch[0].to(rank)
                    input_mask = batch[1].to(rank)
                    label_pos = batch[2].to(rank)
                    match_idx = label_pos >= 0
                    predictions = model(input_ids,
                                        pred_mode="mlm",
                                        token_type_ids=None, 
                                        attention_mask=input_mask)
                    _, sorted_res = torch.topk(predictions[match_idx], top_pred_num, dim=-1)
                    label_idx = label_pos[match_idx]
                    for i, word_list in enumerate(sorted_res):
                        for j, word_id in enumerate(word_list):
                            category_words_freq[label_idx[i].item()][word_id.item()] += 1
            save_file = os.path.join(self.temp_dir, f"{rank}_"+loader_name)
            torch.save(category_words_freq, save_file)
        except RuntimeError as err:
            self.cuda_mem_error(err, "eval", rank)

    # construct category vocabulary
    def category_vocabulary(self, top_pred_num=50, category_vocab_size=100, loader_name="category_vocab.pt", loader_freq_name="category_vocab_freq.pt", check_exist = True):
        loader_file = os.path.join(self.dataset_dir, loader_name)
        print('GPU AVAILABLE : ', torch.cuda.is_available())
        if os.path.exists(loader_file) and check_exist:
            print(f"Loading category vocabulary from {loader_file}")
            self.category_vocab = torch.load(loader_file)
            self.category_words_freq = torch.load(os.path.join(self.dataset_dir, loader_freq_name))
        else:
            print("Contructing category vocabulary.")

            if not os.path.exists(self.temp_dir):
                os.makedirs(self.temp_dir)
            mp.spawn(self.category_vocabulary_dist, nprocs=self.world_size, args=(top_pred_num, loader_name))
            gather_res = []
            for f in os.listdir(self.temp_dir):
                if f[-3:] == '.pt':
                    gather_res.append(torch.load(os.path.join(self.temp_dir, f)))
            assert len(gather_res) == self.world_size, "Number of saved files not equal to number of processes!"
            self.category_words_freq = {i: defaultdict(float) for i in range(self.num_keywords)}
            for i in range(self.num_keywords):
                for category_words_freq in gather_res:
                    for word_id, freq in category_words_freq[i].items():
                        self.category_words_freq[i][word_id] += freq

            print('Loop over cate_vocab :', self.vocab_loop_counter)

            if self.vocab_loop_counter == 0:
                self.old_category_vocab_freq = self.category_words_freq

                for i in range(self.num_keywords):
                    self.label_names_used[i] = self.label_name_dict[i]
                self.vocab_loop_counter += 1
            else:
                
                #### we enter the second loop and needs to average result
                ### But we only run loop over class in self.class_to_loop_over

                for i in self.old_category_vocab_freq:
                    if i in self.class_to_loop_over:
                        j = self.class_to_loop_over.index(i)
                        for word_id, freq in self.old_category_vocab_freq[i].items():
                            if word_id in self.category_words_freq[j].keys() :
                                self.old_category_vocab_freq[i][word_id] += self.category_words_freq[j][word_id]

                        for word_id, freq in self.category_words_freq[j].items():
                            if word_id not in self.old_category_vocab_freq[i].items():
                                self.old_category_vocab_freq[i][word_id] = self.category_words_freq[j][word_id]

                self.category_words_freq = self.old_category_vocab_freq
                
                self.vocab_loop_counter += 1
            self.label_name_dict = self.label_names_used
            self.filter_keywords(category_vocab_size)


            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            for i, category_vocab in self.category_vocab.items():
            ### TO DO : Joffrey Recupérer ca
                print(f"Class {i} category vocabulary: {[self.inv_vocab[w] for w in category_vocab]}\n")

            self.class_to_loop_over = []
            
            for i in self.occurences_per_class:
                if self.occurences_per_class[i] < self.minimum_occurences_per_class:
                    self.class_to_loop_over.append(i)
            if len(self.class_to_loop_over) > 0:
                new_label_names_file = "temporary_label_names.txt"
                labels_to_write = []

                print('Label name already used', self.label_names_used)
                for i, cat_dict in self.category_words_freq.items():
                    if i in self.class_to_loop_over:
                        print("Loop over ", i)
                        new_label_names = sorted(cat_dict.items(), key=lambda item: item[1], reverse=True)  
                        for name, frequency in new_label_names:
                            if self.inv_vocab[name] not in self.label_names_used[i]:
                                if name in self.category_vocab[i]:
                                    new_label_name = name
                                    self.label_names_used[i].append(self.inv_vocab[new_label_name])
                                    print('new label name', self.inv_vocab[new_label_name], frequency)
                                    print('new label name used :', self.label_names_used)
                                    break
                    
                    
                        positive_or_negative = 'positive' if self.label_name_positivity[i]==1 else "negative"
                        labels_to_write.append(self.inv_vocab[new_label_name]+';'+positive_or_negative+'\n')

                with open(os.path.join(self.dataset_dir, "temporary_label_names.txt"), 'w') as file:
                    file.writelines(labels_to_write)
                    del labels_to_write

                ### Read new labels name from the temp file
                self.read_label_names(self.dataset_dir, new_label_names_file,change_positivity_var= False)
                ### Look for occurences of these new labels
                self.read_data(self.args.dataset_dir, self.args.train_file,self.args.train_label_file, self.args.test_file, self.args.test_label_file, check_exist=False)
                new_occurences = self.count_occurences(self.args.dataset_dir,self.args.train_file)
                print('new_occurences', new_occurences)
                for i, class_idx in enumerate(self.class_to_loop_over):
                    self.occurences_per_class[class_idx] += new_occurences[i]
                print('occurences', self.occurences_per_class)
                self.category_vocabulary(check_exist = False)
            try :
                os.remove(os.path.join(self.dataset_dir, "temporary_label_names.txt"))
            except:
                pass
                
            torch.save(self.category_vocab, loader_file)
            torch.save(self.category_words_freq, os.path.join(self.dataset_dir, loader_freq_name))

    # prepare self supervision for masked category prediction (distributed function)
    def prepare_mcp_dist(self, rank, top_pred_num=50, match_threshold=25, loader_name="mcp_train.pt"):
        model = self.set_up_dist(rank)
        model.eval()
        train_dataset_loader = self.make_dataloader(rank, self.train_data, self.eval_batch_size)
        if len(train_dataset_loader.dataset[0]) == 3:
            label_present = True
            print('train label present in the dataset')
        else :
            label_present = False
            print('No ground truth label present in the dataset')
        all_input_ids = []
        all_mask_label = []
        all_input_mask = []
        all_input_labels = []
        category_doc_num = defaultdict(int)
        wrap_train_dataset_loader = tqdm(train_dataset_loader) if rank == 0 else train_dataset_loader
        try:
            for batch in wrap_train_dataset_loader:
                with torch.no_grad():
                    input_ids = batch[0].to(rank)
                    input_mask = batch[1].to(rank)
                    if label_present:
                        input_labels = batch[2].to(rank)
                    predictions = model(input_ids,
                                        pred_mode="mlm",
                                        token_type_ids=None,
                                        attention_mask=input_mask)
                    _, sorted_res = torch.topk(predictions, top_pred_num, dim=-1)
                    for i, category_vocab in self.category_vocab.items():
                        match_idx = torch.zeros_like(sorted_res).bool()
                        for word_id in category_vocab:
                            match_idx = (sorted_res == word_id) | match_idx
                        match_count = torch.sum(match_idx.int(), dim=-1)
                        valid_idx = (match_count > match_threshold) & (input_mask > 0)
                        valid_doc = torch.sum(valid_idx, dim=-1) > 0
                        if valid_doc.any():
                            mask_label = -1 * torch.ones_like(input_ids)
                            mask_label[valid_idx] = i  ## true label = [true_label_keywrd_1, true_label_keyword_2, ...]
                            all_input_ids.append(input_ids[valid_doc].cpu())
                            all_mask_label.append(mask_label[valid_doc].cpu())
                            all_input_mask.append(input_mask[valid_doc].cpu())
                            if label_present:
                                all_input_labels.append(input_labels[valid_doc].cpu())
                            category_doc_num[i] += valid_doc.int().sum().item()
            all_input_ids = torch.cat(all_input_ids, dim=0)
            all_mask_label = torch.cat(all_mask_label, dim=0)
            all_input_mask = torch.cat(all_input_mask, dim=0)
            if label_present:
                all_input_labels = torch.cat(all_input_labels, dim=0)

            save_dict = {
                "all_input_ids": all_input_ids,
                "all_mask_label": all_mask_label,
                "all_input_mask": all_input_mask,
                "category_doc_num": category_doc_num,
                "true_label_class": all_input_labels,
            }
            save_file = os.path.join(self.temp_dir, f"{rank}_"+loader_name)
            torch.save(save_dict, save_file)
        except RuntimeError as err:
            self.cuda_mem_error(err, "eval", rank)

    # prepare self supervision for masked category prediction
    def prepare_mcp(self, top_pred_num=50, match_threshold=25, loader_name="mcp_train.pt"):
        loader_file = os.path.join(self.dataset_dir, loader_name)
        if os.path.exists(loader_file):
            print(f"Loading masked category prediction data from {loader_file}")
            self.mcp_data = torch.load(loader_file)
        else:
            loader_file = os.path.join(self.dataset_dir, loader_name)
            print("Preparing self supervision for masked category prediction.")
            if not os.path.exists(self.temp_dir):
                os.makedirs(self.temp_dir)
            mp.spawn(self.prepare_mcp_dist, nprocs=self.world_size, args=(top_pred_num, match_threshold, loader_name))
            gather_res = []
            for f in os.listdir(self.temp_dir):
                if f[-3:] == '.pt':
                    gather_res.append(torch.load(os.path.join(self.temp_dir, f)))
            assert len(gather_res) == self.world_size, "Number of saved files not equal to number of processes!"
            all_input_ids = torch.cat([res["all_input_ids"] for res in gather_res], dim=0)
            all_mask_label = torch.cat([res["all_mask_label"] for res in gather_res], dim=0)
            all_input_mask = torch.cat([res["all_input_mask"] for res in gather_res], dim=0)
            category_doc_num = {i: 0 for i in range(self.num_keywords)}
            try:
                ground_truth_label = torch.cat([res['true_label_class'] for res in gather_res], dim=0)
            except:
                print("No ground_truth given")
            for i in category_doc_num:
                for res in gather_res:
                    if i in res["category_doc_num"]:
                        category_doc_num[i] += res["category_doc_num"][i]
            print(f"Number of documents with category indicative terms found for each category is: {category_doc_num}")
            self.mcp_data = {"input_ids": all_input_ids, "attention_masks": all_input_mask, "labels": all_mask_label, "ground_truth": ground_truth_label}
            torch.save(self.mcp_data, loader_file)
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            for i in category_doc_num:
                assert category_doc_num[i] > 10, f"Too few ({category_doc_num[i]}) documents with category indicative terms found for category {i}; " \
                       "try to add more unlabeled documents to the training corpus (recommend) or reduce `--match_threshold` (not recommend)"
        ### TO DO : Joffrey Recupérer ca
        print(f"There are totally {len(self.mcp_data['input_ids'])} documents with category indicative terms.")



    def training_set_statistics(self, positive_label = None, negative_label = None, loader_name = "mcp_train.pt"):
        print('Computing statistics Positive set')
    
        if positive_label is None:
            positive_label = self.true_label
        if negative_label is None :
            negative_label = self.negative_keywords
        assert(len(positive_label)>0)
        loader_file = os.path.join(self.dataset_dir, loader_name)
        if os.path.exists(loader_file):
            print(f"Loading masked category prediction data from {loader_file}")
            self.mcp_data = torch.load(loader_file)
        else: 
            self.prepare_mcp()
            self.mcp_data = torch.load(loader_file)
        
        ### Get an assumed label per text
        assumed_label = []
        for x in self.mcp_data['labels']:
            # if all assumed labels are the same accross words of the texts
            if len(np.unique(x[x!=-1].numpy())) == 1:       
                assumed_label.append(np.unique(x[x!=-1].numpy())[0])
            else:
                assumed_label.append(-1)
        self.mcp_data['assumed_labels'] = assumed_label
        assumed_label = np.array(assumed_label)
        ground = self.mcp_data['ground_truth'].numpy()
        df = pd.DataFrame([ground, assumed_label]).T
        df.columns = ['ground','assumed']
        
        number_of_true_positive = len(df[(df['ground'].isin(positive_label)) & (df['assumed'].apply(lambda x: self.label_name_positivity[x]))])
        number_of_false_positive = len(df[(~df['ground'].isin(positive_label)) & (df['assumed'].apply(lambda x: self.label_name_positivity[x]))])

        if len(negative_label) > 0 : 
            number_of_true_negative = len(df[(df['ground'].isin(negative_label)) & (df['assumed'].apply(lambda x: self.label_name_positivity[x]==0))])
            number_of_false_negative = len(df[(~df['ground'].isin(negative_label)) & (df['assumed'].apply(lambda x: self.label_name_positivity[x]==0))])

        self.pos_set_accuracy = number_of_true_positive/(number_of_true_positive+number_of_false_positive)
        self.number_elements_pos_set = number_of_true_positive+number_of_false_positive

        ### TO DO : Joffrey Recupérer ca
        if self.verbose :
            print("PRECISION of the positive set : ", number_of_true_positive/(number_of_true_positive+number_of_false_positive))
            print("NUMBER OF ELEMENTS in the positive set", self.number_elements_pos_set)
            if len(negative_label) > 0 : 
                print("Precision of the negative set : ", number_of_true_negative/(number_of_true_negative+number_of_false_negative))
                print("NUMBER OF ELEMENTS in the negative set", number_of_true_negative+number_of_false_negative)



        # data_vocab = torch.load(os.path.join(self.dataset_dir, "category_vocab.pt"))
        # label_data = torch.load(os.path.join(self.dataset_dir, "label_name_data.pt"))
        # train_data = torch.load(os.path.join(self.dataset_dir, "train.pt"))

        # corpus = open(os.path.join(self.dataset_dir,'train.txt'), encoding="utf-8")
        # true_labels = open(os.path.join(self.dataset_dir,'train_labels.txt'), encoding="utf-8")
        # docs_labels = [int(doc.strip()) for doc in true_labels.readlines()]
        # self.docs_labels = docs_labels
        # dict_label = {0:[], 1:[], 2:[],3:[]}


        # for i, label in enumerate(docs_labels):
        #     dict_label[int(label)].append(i)
        # docs = [doc.strip() for doc in corpus.readlines()]

        # self.train_docs = docs

    def loading_for_test(self, loader_name='train.txt', loader_label_name = 'train_labels.txt'):
        corpus = open(os.path.join(self.dataset_dir,loader_name), encoding="utf-8")
        docs = [doc.strip() for doc in corpus.readlines()]
        true_labels = open(os.path.join(self.dataset_dir,loader_label_name), encoding="utf-8")
        docs_labels = [int(doc.strip()) for doc in true_labels.readlines()]
        return docs, docs_labels

    def test(self, model,test_data = 'test.txt',test_data_labels = 'test_labels.txt', number = 1024, test_batch_size = 32, all = False, binary = True, true_label = None, device = None):
        if true_label is None:
            assert(self.true_label is not None)
            true_label = self.true_label
        docs, docs_labels = self.loading_for_test(test_data, test_data_labels)
        if device is None:
            device = self.device
        model.eval()

        old_true_positive = 0

        true_negative = 0
        true_positive = 0
        false_positive = 0
        false_negative = 0
        correct_pred = 0
        negative = 0
        divider = number
        if all:
            test_list = list(range(len(docs)))
            divider = len(docs)
        else:
            test_list = random.sample(list(range(len(docs))), k = number)


            
        inputs = torch.stack([self.non_batch_encode(docs[i])[0].squeeze() for i in test_list])
        attention_mask = torch.stack([self.non_batch_encode(docs[i])[1].squeeze() for i in test_list])
        true_labels = torch.stack([torch.tensor(int(docs_labels[i] in self.true_label)) for i in test_list])
        

        test_dataset = TensorDataset(inputs, attention_mask, true_labels)
        test_dataloader = DataLoader(test_dataset, batch_size = test_batch_size)
        with torch.no_grad():
            for n, batch in enumerate(test_dataloader):
                inputs_test, attention_test, labels_test = batch
                logits = model(inputs_test.to(device),attention_mask=attention_test.to(device), pred_mode='classification')
                logits_cls = logits[:,0]
                prediction = torch.argmax(logits_cls, -1)

                if binary:

                    predictions = torch.tensor([self.label_name_positivity[x.cpu().item()] for x in prediction])
                    true_positive += (predictions*labels_test).sum().item()
                    true_negative += ((1-predictions.cpu())*(1-labels_test)).sum().item()
                    false_positive += ((predictions.cpu())*(1-labels_test)).sum().item()
                    false_negative += ((1-predictions.cpu())*(labels_test)).sum().item()
                    correct_pred += (labels_test == predictions.cpu()).sum().item()


                else:
                    true_positive += (prediction.cpu()*labels_test).sum().item()
                    true_negative += ((1-prediction.cpu())*(1-labels_test)).sum().item()
                    false_positive += ((prediction.cpu())*(1-labels_test)).sum().item()
                    false_negative += ((1-prediction.cpu())*(labels_test)).sum().item()
                    correct_pred += (labels_test == prediction.cpu()).sum().item()
                assert (correct_pred == (true_positive + true_negative))

            assert(true_positive+true_negative+false_positive+false_negative == divider)
            accuracy = correct_pred / divider
            
        if (true_positive+false_positive) > 0:
            precision = true_positive / (true_positive+false_positive)
            print('Precision', precision)
        else : 
            precision = None
            print("Precision Undefined")
        if (true_positive+false_negative) > 0 :
            recall = true_positive/(true_positive+false_negative)
            print('Recall', recall)
        else :
            recall = None
            print("Recall Undefined")
        if (recall is not None) and (precision is not None) :
            if recall + precision > 0 :
                f1_score = 2*(recall*precision)/(recall+precision)
                print("F1_score", f1_score)
            else:
                f1_score = None
                print("F1_score Undefined")
        else:
            f1_score = None
            print("F1_score Undefined")
        print("Accuracy ", accuracy)
        model.train()
        return accuracy, precision, recall, f1_score

    def return_all_keywords_related_vocab(self, positive = True, negative = False, loader_name = "category_vocab.pt"):
        '''
        The function takes as input the category vocab built around the different keywords , the type of keywords one is interested in 'positive', 'negative',
        or both, and return a list with all the related words from this type of keywords.
        Example:
        return_all_keywords_related_vocab(positive) will return the list of concatenated category vocabs of all positive keywords(tokens and strings so 2 lists)
        '''
        list_all_related_words = []
        list_all_related_words_tokens = []
        data_vocab = torch.load(os.path.join(self.dataset_dir, loader_name))
        if positive : 
            for k in data_vocab.keys():
                if self.label_name_positivity[k] :
                    list_all_related_words_tokens += list(data_vocab[k])
                    list_all_related_words += [self.inv_vocab[w] for w in data_vocab[k]]
        if negative:
            for k in data_vocab.keys():
                if ~self.label_name_positivity[k] :
                    list_all_related_words_tokens += list(data_vocab[k])
                    list_all_related_words += [self.inv_vocab[w] for w in data_vocab[k]]

        return list_all_related_words_tokens, list_all_related_words

    def joint_cate_vocab(self, positive = True, negative = False, topk = 100, min_occurences = 20,
                        loader_name='category_vocab.pt', loader_name_freq = 'category_vocab_freq.pt'):
        '''
        The function a joint category vocab(a list of words in string, and their tokenized version) of topk words in the positive, negative or both type of keywords related vocab.
        '''

        category_vocab_freq = torch.load(os.path.join(self.dataset_dir, loader_name_freq))
        category_vocab = torch.load(os.path.join(self.dataset_dir, loader_name))

        new_category_vocab_freq = {i : {j : category_vocab_freq[i][j] for j in category_vocab_freq[i] if 
                                j in category_vocab[i]} for i in category_vocab_freq}
        
        df = pd.DataFrame(new_category_vocab_freq).fillna(0)
        if positive :
            if negative :
                joint_vocab_tokens = list(df.sum(1)[df.sum(1)>min_occurences].sort_values(ascending = False).head(topk).index)
            else :
                joint_vocab_tokens = list(df[[i for i in df.columns if self.label_name_positivity[i]]].sum(1)[df[[i for i in df.columns if self.label_name_positivity[i]]].sum(1)>min_occurences].sort_values(ascending = False).head(topk).index)
        elif negative:
            joint_vocab_tokens = list(df[[i for i in df.columns if ~self.label_name_positivity[i]]].sum(1)[df[[i for i in df.columns if ~self.label_name_positivity[i]]].sum(1)>min_occurences].sort_values(ascending = False).head(topk).index)
        else :
            print('Should ask for either positive, negative, or both')
        
        joint_vocab = [self.inv_vocab[w] for w in joint_vocab_tokens]

        return joint_vocab_tokens, joint_vocab


    def compute_preset_negative(self,docs = None, verbose = None, loader_name = 'pre_negative_dataloader.pt'):
        print("Computing preset of possible negative texts")
        loader_file = os.path.join(self.dataset_dir,loader_name)
        if os.path.exists(loader_file):
            print(f"Loading pre_negative_dataloader data from {loader_file}")
            self.pre_negative_dataloader = torch.load(loader_file)
        else:
            negative_doc=[]
            negative_doc_label = []

            list_all_positive_words_tokens, list_all_positive_words = self.joint_cate_vocab(positive = True, negative = False, min_occurences=150)
            if docs is None:
                try:
                    docs = self.train_docs
                except AttributeError:
                    corpus = open(os.path.join(self.dataset_dir, self.train_text_file), encoding="utf-8")
                    docs = [doc.strip() for doc in corpus.readlines()]
                    self.train_docs = docs
                docs = self.train_docs
            if verbose is None:
                verbose = self.verbose
            ### Selecting texts without positive keywords
            for k, doc in tqdm(enumerate(docs)):
                tokenized_doc = self.tokenizer.tokenize(doc)
                new_doc = []
                wordpcs = []
                label_idx = -1 * torch.ones(512, dtype=torch.long)
                for idx, wordpc in enumerate(tokenized_doc):
                    wordpcs.append(wordpc[2:] if wordpc.startswith("##") else wordpc)
                    if idx >= 512 - 1: # last index will be [SEP] token
                        break
                    if idx == len(doc) - 1 or not doc[idx+1].startswith("##"):
                        word = ''.join(wordpcs)
                        if word in list_all_positive_words:
                            label_idx[idx] = 0
                            break
                        new_word = ''.join(wordpcs)
                        if new_word != self.tokenizer.unk_token:
                            idx += len(wordpcs)
                            new_doc.append(new_word)
                        wordpcs = []
                if (label_idx>=0).any():
                    continue
                else:
                    negative_doc_label.append(self.docs_labels[k])
                    negative_doc.append(doc)
            if verbose : 
                print("Negative pre-set", len(negative_doc))
                print("Precision pre-set, ", len([k for k in negative_doc_label if k not in self.true_label])/len(negative_doc_label))
            self.pre_negative_docs = negative_doc
            self.pre_negative_docs_label = negative_doc_label

            ### Formating into tensors
            inputs_list = []
            masks_list = []
            for doc in tqdm(negative_doc):
                input_ids, input_mask = self.non_batch_encode(doc)
                inputs_list.append(input_ids)
                masks_list.append(input_mask)
            input_tensor = torch.stack(inputs_list).squeeze()
            mask_tensor = torch.stack(masks_list).squeeze()
            label_tensor = torch.stack([torch.tensor(i).unsqueeze(0) for i in negative_doc_label])
            dataset = torch.utils.data.TensorDataset(input_tensor,mask_tensor, label_tensor)
            dataloader = torch.utils.data.DataLoader(dataset, shuffle = False, batch_size = 8)
            self.pre_negative_dataset = dataset
            self.pre_negative_dataloader = dataloader

            ### Export
            torch.save(self.pre_negative_dataloader,os.path.join(self.dataset_dir,loader_name))
        

    def compute_set_negative(self, model = None, relative_factor_pos_neg = 3, early_stopping = True, topk = 20, min_similar_words = 0,
                                max_category_word = 1, verbose = None, device = None, 
                                loader_name = 'pre_negative_dataloader.pt', loader_cate_name='mcp_train.pt'):

        print('Compute Negative Training Set')
        if device is None:
            device = self.device
        if model is None: 
            model = self.model
        if verbose is None:
            verbose = self.verbose
        model.to(device)

        self.compute_preset_negative(loader_name=loader_name)
            # self.pre_negative_dataloader = torch.load(loader_file)
        if os.path.exists(os.path.join(self.dataset_dir,'negative_dataset.pt')):
            print('Loading results')
            self.negative_dataset = torch.load(os.path.join(self.dataset_dir,'negative_dataset.pt'))
        else:
            # Parameters controlling the size of the neg set
            relative_factor_pos_neg = relative_factor_pos_neg
            early_stopping = True

            verified_negative = []
            ground_label = []
            correct_label = 0
            if verbose is None:
                verbose = self.verbose
            
            topk = topk
            min_similar_words = min_similar_words
            max_category_word = max_category_word
            list_positive_words_tokens, list_positive_words = self.joint_cate_vocab(positive = True, min_occurences = 100)
            print('list_positive_words',list_positive_words)
            # Computations
            with torch.no_grad():
                for k, batch in tqdm(enumerate(self.pre_negative_dataloader)):

                    input_ids, input_mask, label_id = batch
                    predictions = model(input_ids.to(device),
                                    pred_mode="mlm",
                                    token_type_ids=None, 
                                    attention_mask=input_mask.to(device))
                    # Loop over the documents in the batch
                    for i, doc in enumerate(predictions.cpu()):
                        # Selecting only the position corresponding to a word not the PADDING
                        masked_pred = doc[:input_mask[i].sum().item(),:]
                        # Selecting the TOP 'k' words predicted at each position
                        _ , words = torch.topk(masked_pred, topk, -1)
                        valid =  True
                        counter = 0
                        # Loop over the words in each document
                        for word in words.squeeze():
                            
                            counter += int(len(np.intersect1d(word.cpu().numpy(),list_positive_words_tokens))>min_similar_words)
                            if counter > max_category_word:
                                valid = False
                                break
                        if valid:            
                            verified_negative.append(k*4+i)
                            ground_label.append(label_id[i])
                            if label_id[i] not in self.true_label:
                                correct_label += 1 
                    if (len(verified_negative) > relative_factor_pos_neg * self.number_elements_pos_set) and early_stopping:
                        break
                    if len(verified_negative) > 0:
                        neg_set_accuracy = correct_label/len(verified_negative)
                    else:
                        neg_set_accuracy = 0

                    ### TO DO : Joffrey Recupérer ca
                    if k%100 == 0 and verbose:
                        if len(verified_negative)>0:
                            print('accuracy :', neg_set_accuracy)
                            print('number of elements retrieved', len(verified_negative))
                
            self.negative_dataset = Subset(self.pre_negative_dataloader.dataset, verified_negative)
            
            # Keep only the ones not in a negative_keyword class if this class has minimum number of texts
            if len(self.negative_dataset) > 250:
                

                if 0 in self.label_name_positivity.values():
                    valid_indices = []
                    original_length = len(self.negative_dataset)
                    dataset = torch.load(os.path.join(self.dataset_dir,loader_cate_name))
                    dataset = dataset['input_ids']
                    for i, neg in enumerate(self.negative_dataset):
                        valid = True
                        for data in dataset:
                            if (neg[0]==data).all().item():
                                valid = False
                                break
                        if valid:
                            valid_indices.append(i)

                    self.negative_dataset = Subset(self.negative_dataset, valid_indices)
                print("Difference after intersection reduction :", original_length - len(self.negative_dataset))
                #EXPORT THE DATASET 
                self.negative_dataset = Subset(self.pre_negative_dataloader.dataset, verified_negative)
                torch.save(self.negative_dataset, os.path.join(self.dataset_dir,'negative_dataset.pt'))
            else :
                self.negative_dataset = self.pre_negative_dataloader.dataset
                self.look_for_negative = False
       


    def train(self, model = None, batch_size = None, accum_steps = 8, epochs = 3, loader_positive = 'mcp_train.pt', 
                loader_negative = 'negative_dataset.pt',weighted_sampler = True,  device = None):
        print('Training ...')

        if model is None:
            model = self.model
        if device is None:
            device = self.device
        if batch_size is None:
            batch_size = self.train_batch_size

        positive_loader_file = os.path.join(self.dataset_dir, loader_positive)
        if os.path.exists(positive_loader_file):
            print(f"Loading masked category prediction data from {positive_loader_file}")
            mcp_data = torch.load(positive_loader_file)
        else: 
            self.prepare_mcp()
            mcp_data = torch.load(positive_loader_file)
        

        assumed_label = []
        for x in self.mcp_data['labels']:
        # if all assumed labels are the same accross words of the texts
            if len(np.unique(x[x!=-1].numpy())) == 1:       
                assumed_label.append(np.unique(x[x!=-1].numpy())[0])
            else:
                assumed_label.append(-1)
                
        self.mcp_data['assumed_labels'] = assumed_label
        ### GET RID OF THE TEXT WHERE MULTIPLE LABELS WHERE FOUND ; Possible improvements
        index_select = [i  for i, x in enumerate(assumed_label) if x!=-1]
        

        inputs = mcp_data['input_ids'].tolist()
        inputs_ids = torch.stack([torch.tensor(inputs[i]) for i in index_select])

        attention = mcp_data['attention_masks'].tolist()
        attention_masks = torch.stack([torch.tensor(attention[i]) for i in index_select])

        all_labels = self.mcp_data['assumed_labels']
        
        labels_for_training = torch.stack([torch.tensor(all_labels[i]) for i in index_select])
        print(labels_for_training)
        self.positive_dataset = torch.utils.data.TensorDataset(inputs_ids, attention_masks, labels_for_training)


        ### Negative
        
        negative_loader_file = os.path.join(self.dataset_dir, loader_negative)

        if os.path.exists(negative_loader_file):
            self.negative_dataset = torch.load(negative_loader_file)
        elif self.look_for_negative:
            self.compute_set_negative()
        else :
            pass


        if (self.look_for_negative) or (0 not in self.label_name_positivity.values()):
        ### CONSTRUCTION OF THE DATALOADER
            data = torch.stack([negative_data[0][:self.max_len] for negative_data in self.negative_dataset] + 
                        [positive_data[0][:self.max_len] for positive_data in self.positive_dataset])
            ### New negative class always the last class
            mask = torch.stack([negative_data[1][:self.max_len] for negative_data in self.negative_dataset] + 
                        [positive_data[1][:self.max_len] for positive_data in self.positive_dataset])


            target = np.hstack(((np.zeros(int(len(self.negative_dataset)), dtype=np.int32)+len(self.label_name_positivity)-1),
                    np.array([keyword_data[2] for keyword_data in self.positive_dataset])))


            target = torch.from_numpy(np.expand_dims(target, 1)).long()


        else: ### in case the negative set construction hasnt been great and there is already negative keyword then just base everything on keyword
        ### CONSTRUCTION OF THE DATALOADER
            data = torch.stack([positive_data[0][:self.max_len] for positive_data in self.positive_dataset])
            ### New negative class always the last class
            mask = torch.stack([positive_data[1][:self.max_len] for positive_data in self.positive_dataset])


            target = np.array([keyword_data[2] for keyword_data in self.positive_dataset])
            labels_present = []
            for i in np.unique(target):
                labels_present.append(self.label_name_positivity[i])
            assert((1 in labels_present) and (0 in labels_present))
            target = torch.from_numpy(np.expand_dims(target, 1)).long()
            self.num_class = self.num_class -1
            ## Redefine the model with one fewer outcome
            self.model = ClassifModel.from_pretrained(self.pretrained_lm,
                                                   output_attentions=False,
                                                   output_hidden_states=False,
                                                   num_labels=self.num_class).to(device)
            model = self.model

        train_dataset = torch.utils.data.TensorDataset(data,mask, target)
        ### Construction of the weighted sampler based on sets' sizes and of the target vector #########

        if weighted_sampler:

            class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
            weight = 1. / class_sample_count
            samples_weight = np.array([weight[t] for t in target])
            samples_weight = torch.from_numpy(samples_weight)
            samples_weigth = samples_weight.double()
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
            train_loader = DataLoader(train_dataset, batch_size = batch_size, sampler=sampler)

        else:

            train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)



        ### TRAINING HYPERPARAMETERS AND PARAMETERS

        accum_steps = accum_steps
        epochs = epochs
        learning_rate = 5e-5
        train_loss = nn.CrossEntropyLoss()
        total_steps = len(train_loader) * epochs / accum_steps
        number_of_mask = 2
        assert((accum_steps*batch_size)>=128)
        print(type(mcp_data['labels']))
        pos_set_accuracy = np.sum(mcp_data['labels'].numpy())/len(mcp_data['labels'])
        neg_set_accuracy = 0 # TO DO


        parameters = {'epochs':epochs, 'learning_rate':learning_rate, 'number_of_mask':number_of_mask,
            'accum_steps':accum_steps, 
            'batch_size': batch_size, 
            'pos_set' : len(self.positive_dataset),
            'pos_set_accuracy' : pos_set_accuracy,
            'neg_set' : len(self.negative_dataset),
            'neg_set_accuracy' :neg_set_accuracy,
            'min occurences' : self.minimum_occurences_per_class}

        optimizer = AdamW([{'params' : filter(lambda p: p.requires_grad, self.model.bert.parameters()), 'lr' : 1e-2*learning_rate}, 
                            {'params' : filter(lambda p: p.requires_grad, self.model.classifier.parameters()), 'lr' : learning_rate}], eps=1e-8)
        # optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*total_steps, num_training_steps=total_steps)


        number_of_mask = 1 
        # Metrics
        losses_track = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        global_steps = []

        try:
            for i in range(epochs):
                self.model.train()
                total_train_loss = 0
                self.model.zero_grad()
                print('Epoch : ', i)
                for j, batch in enumerate(train_loader):
                    input_ids = batch[0].to(device)
                    input_mask = batch[1].to(device)
                    labels = batch[2].to(device)


                    ### RANDOM MASKING
                    random_masking = random.choices(list(range(199)),k = number_of_mask * input_ids.size(0))
                    for i, mask_pos in enumerate(random_masking):
                        input_ids[i%input_ids.size(0),mask_pos+1] = self.tokenizer.get_vocab()[self.tokenizer.mask_token]
                    
                    ### PREDICTION
                    logits = self.model(input_ids, 
                                pred_mode="classification",
                                token_type_ids=None, 
                                attention_mask=input_mask)
                    ### LOSS
                    logits_cls = logits[:,0]
                    loss = train_loss(logits_cls.contiguous().view(-1, self.num_class), labels.contiguous().view(-1)) / accum_steps            
                    total_train_loss += loss.item()
                    loss.backward()
                    if (j+1) % accum_steps == 0:
                        # Clip the norm of the gradients to 1.0.
                        
                        losses_track.append(loss*accum_steps)
                        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        self.model.zero_grad()
                        
                    ### TO DO : Joffrey Recupérer ca
                    if j % (3*accum_steps) == 0 :
                        print('loss',loss*accum_steps)
                        accuracy, precision, recall, f1_score = self.test(self.model, number = 1024)
                        accuracies.append(accuracy)
                        precisions.append(precision)
                        recalls.append(recall)
                        f1_scores.append(f1_score)
                        global_steps.append(j)
                avg_train_loss = torch.tensor([total_train_loss / len(train_loader) * accum_steps]).to(device)
                print(f"Average training loss: {avg_train_loss.mean().item()}")

        except RuntimeError as err:
            print(err)

        ### Export
        p.dump(accuracies,open(os.path.join(self.dataset_dir,'accuracy.p'),'wb'))
        p.dump(losses_track,open(os.path.join(self.dataset_dir,'loss.p'),'wb'))
        p.dump(precisions,open(os.path.join(self.dataset_dir,'precision.p'),'wb'))
        p.dump(recalls,open(os.path.join(self.dataset_dir,'recall.p'),'wb'))
        p.dump(f1_scores,open(os.path.join(self.dataset_dir,'f1_score.p'),'wb'))
        p.dump(parameters, open(os.path.join(self.dataset_dir,'parameters.p'), 'wb'))
        torch.save(self.model.state_dict(), os.path.join(self.dataset_dir,'model.pt'))



    # prepare self training data and target distribution
    def prepare_self_train_data(self, rank, model, idx):

        ### TO DO : ADD UNCERTAINTY WITH DROPOUT
        target_num = min(self.world_size * self.train_batch_size * self.update_interval * self.accum_steps, len(self.train_data["input_ids"]))
        if idx + target_num >= len(self.train_data["input_ids"]):
            select_idx = torch.cat((torch.arange(idx, len(self.train_data["input_ids"])),
                                    torch.arange(idx + target_num - len(self.train_data["input_ids"]))))
        else:
            select_idx = torch.arange(idx, idx + target_num)
        assert len(select_idx) == target_num
        idx = (idx + len(select_idx)) % len(self.train_data["input_ids"])
        select_dataset = {"input_ids": self.train_data["input_ids"][select_idx],
                          "attention_masks": self.train_data["attention_masks"][select_idx]}
        dataset_loader = self.make_dataloader(rank, select_dataset, self.eval_batch_size)
        input_ids, input_mask, preds = self.inference(model, dataset_loader, rank, return_type="data")
        gather_input_ids = [torch.ones_like(input_ids) for _ in range(self.world_size)]
        gather_input_mask = [torch.ones_like(input_mask) for _ in range(self.world_size)]
        gather_preds = [torch.ones_like(preds) for _ in range(self.world_size)]
        dist.all_gather(gather_input_ids, input_ids)
        dist.all_gather(gather_input_mask, input_mask)
        dist.all_gather(gather_preds, preds)
        input_ids = torch.cat(gather_input_ids, dim=0).cpu()
        input_mask = torch.cat(gather_input_mask, dim=0).cpu()
        all_preds = torch.cat(gather_preds, dim=0).cpu()
        weight = all_preds**2 / torch.sum(all_preds, dim=0)
        target_dist = (weight.t() / torch.sum(weight, dim=1)).t()
        all_target_pred = target_dist.argmax(dim=-1)
        agree = (all_preds.argmax(dim=-1) == all_target_pred).int().sum().item() / len(all_target_pred)
        self_train_dict = {"input_ids": input_ids, "attention_masks": input_mask, "labels": target_dist}
        return self_train_dict, idx, agree

    # train a model on batches of data with target labels
    def self_train_batches(self, rank, model, self_train_loader, optimizer, scheduler, test_dataset_loader):


               # Metrics
        losses_track = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        global_steps = []

        model.train()
        total_train_loss = 0
        wrap_train_dataset_loader = tqdm(self_train_loader) if rank == 0 else self_train_loader
        model.zero_grad()
        try:
            for j, batch in enumerate(wrap_train_dataset_loader):
                input_ids = batch[0].to(rank)
                input_mask = batch[1].to(rank)
                target_dist = batch[2].to(rank)
                logits = model(input_ids,
                               pred_mode="classification",
                               token_type_ids=None,
                               attention_mask=input_mask)
                logits = logits[:, 0, :]
                preds = nn.LogSoftmax(dim=-1)(logits)
                loss = self.st_loss(preds.view(-1, self.num_class), target_dist.view(-1, self.num_class)) / self.accum_steps
                total_train_loss += loss.item()
                loss.backward()
                if (j+1) % self.accum_steps == 0:
                    # Clip the norm of the gradients to 1.0.
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
            # if self.with_test_label:
            #     acc = self.inference(model, test_dataset_loader, rank, return_type="acc")
            #     gather_acc = [torch.ones_like(acc) for _ in range(self.world_size)]
            #     dist.all_gather(gather_acc, acc)
            #     acc = torch.tensor(gather_acc).mean().item()




            avg_train_loss = torch.tensor([total_train_loss / len(wrap_train_dataset_loader) * self.accum_steps]).to(rank)
            gather_list = [torch.ones_like(avg_train_loss) for _ in range(self.world_size)]
            dist.all_gather(gather_list, avg_train_loss)
            avg_train_loss = torch.tensor(gather_list)
            if rank == 0:
                print(f"lr: {optimizer.param_groups[0]['lr']:.4g}")
                print(f"Average training loss: {avg_train_loss.mean().item()}")
                ### TEST SCORE
                accuracy, precision, recall, f1_score = self.test(model, number = 1024)
                losses_track.append(loss*self.accum_steps)
                accuracies.append(accuracy)
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1_score)
                global_steps.append(j)
                # if self.with_test_label:
                #     print(f"Test acc: {acc}")
        except RuntimeError as err:
            self.cuda_mem_error(err, "train", rank)

    # self training (distributed function)
    def self_train_dist(self, rank, epochs, loader_name="final_model.pt"):
        model = self.set_up_dist(rank)
        test_dataset_loader = self.make_dataloader(rank, self.test_data, self.eval_batch_size) if self.with_test_label else None
        total_steps = int(len(self.train_data["input_ids"]) * epochs / (self.world_size * self.train_batch_size * self.accum_steps))
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-6, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*total_steps, num_training_steps=total_steps)
        idx = 0
        if self.early_stop:
            agree_count = 0
        for i in range(int(total_steps / self.update_interval)):
            self_train_dict, idx, agree = self.prepare_self_train_data(rank, model, idx)
            # early stop if current prediction agrees with target distribution for 3 consecutive updates
            if self.early_stop:
                if 1 - agree < 1e-3:
                    agree_count += 1
                else:
                    agree_count = 0
                if agree_count >= 3:
                    break
            self_train_dataset_loader = self.make_dataloader(rank, self_train_dict, self.train_batch_size)
            self.self_train_batches(rank, model, self_train_dataset_loader, optimizer, scheduler, test_dataset_loader)
        if rank == 0:
            loader_file = os.path.join(self.dataset_dir, loader_name)
            print(f"Saving final model to {loader_file}")
            torch.save(model.module.state_dict(), loader_file)

    # self training
    def self_train(self, epochs, loader_name="final_model.pt"):
        loader_file = os.path.join(self.dataset_dir, loader_name)
        if os.path.exists(loader_file):
            print(f"\nFinal model {loader_file} found, skip self-training")
        else:
            rand_idx = torch.randperm(len(self.train_data["input_ids"]))
            if len(self.train_data) > 2:
                self.train_data = {"input_ids": self.train_data["input_ids"][rand_idx],
                                "attention_masks": self.train_data["attention_masks"][rand_idx],
                                "labels" : self.train_data["labels"][rand_idx]}
            else:
                print('No groud truth label in dataset')
                self.train_data = {"input_ids": self.train_data["input_ids"][rand_idx],
                                "attention_masks": self.train_data["attention_masks"][rand_idx]}
            print(f"\nStart self-training.")
            print('world_size', self.world_size)
            mp.spawn(self.self_train_dist, nprocs=self.world_size, args=(epochs, loader_name))
    
    # use a model to do inference on a dataloader
    def inference(self, model, dataset_loader, rank, return_type):
        if return_type == "data":
            all_input_ids = []
            all_input_mask = []
            all_preds = []
        elif return_type == "acc":
            pred_labels = []
            truth_labels = []
        elif return_type == "pred":
            pred_labels = []
        model.eval()
        try:
            for batch in dataset_loader:
                with torch.no_grad():
                    input_ids = batch[0].to(rank)
                    input_mask = batch[1].to(rank)
                    logits = model(input_ids,
                                   pred_mode="classification",
                                   token_type_ids=None,
                                   attention_mask=input_mask)
                    logits = logits[:, 0, :]
                    if return_type == "data":
                        all_input_ids.append(input_ids)
                        all_input_mask.append(input_mask)
                        all_preds.append(nn.Softmax(dim=-1)(logits))
                    elif return_type == "acc":
                        labels = batch[2]
                        pred_labels.append(torch.argmax(logits, dim=-1).cpu())
                        truth_labels.append(labels)
                    elif return_type == "pred":
                        pred_labels.append(torch.argmax(logits, dim=-1).cpu())
            if return_type == "data":
                all_input_ids = torch.cat(all_input_ids, dim=0)
                all_input_mask = torch.cat(all_input_mask, dim=0)
                all_preds = torch.cat(all_preds, dim=0)
                return all_input_ids, all_input_mask, all_preds
            elif return_type == "acc":
                pred_labels = torch.cat(pred_labels, dim=0)
                truth_labels = torch.cat(truth_labels, dim=0)
                samples = len(truth_labels)
                acc = (pred_labels == truth_labels).float().sum() / samples
                return acc.to(rank)
            elif return_type == "pred":
                pred_labels = torch.cat(pred_labels, dim=0)
                return pred_labels
        except RuntimeError as err:
            self.cuda_mem_error(err, "eval", rank)
    
    # use trained model to make predictions on the test set
    def write_results(self, loader_name="final_model.pt", out_file="out.txt"):
        loader_file = os.path.join(self.dataset_dir, loader_name)
        assert os.path.exists(loader_file)
        print(f"\nLoading model trained via masked category prediction from {loader_file}")
        self.model.load_state_dict(torch.load(loader_file))
        self.model.to(0)
        test_set = TensorDataset(self.test_data["input_ids"], self.test_data["attention_masks"])
        test_dataset_loader = DataLoader(test_set, sampler=SequentialSampler(test_set), batch_size=self.eval_batch_size)
        pred_labels = self.inference(self.model, test_dataset_loader, 0, return_type="pred")
        out_file = os.path.join(self.dataset_dir, out_file)
        print(f"Writing prediction results to {out_file}")
        f_out = open(out_file, 'w')
        for label in pred_labels:
            f_out.write(str(label.item()) + '\n')

    # print error message based on CUDA memory error
    def cuda_mem_error(self, err, mode, rank):
        if rank == 0:
            print(err)
            if "CUDA out of memory" in str(err):
                if mode == "eval":
                    print(f"Your GPUs can't hold the current batch size for evaluation, try to reduce `--eval_batch_size`, current: {self.eval_batch_size}")
                else:
                    print(f"Your GPUs can't hold the current batch size for training, try to reduce `--train_batch_size`, current: {self.train_batch_size}")
        sys.exit(1)
