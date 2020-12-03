#!/usr/bin/env python
# coding: utf-8
## Creating the folder
import warnings
warnings.filterwarnings("ignore")
import os    
import argparse


def main():
    parser = argparse.ArgumentParser(description='main',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--keyword', default='politics',
                        help='keyword')
    parser.add_argument('--number_of_loop', default=1,
                        help='Number of loop to build the category vocabulary')
    parser.add_argument('--dataset_dir', default='agnews', 
			help='Name of the directory where the datasets are stored.')
    args = parser.parse_args()
    directory = args.dataset_dir
    keyword = args.keyword
    number_of_loop_over_vocab = args.number_of_loop

    if keyword == 'company':
        positive_label = [0]
    elif (keyword == 'school') or (keyword == 'university'):
        positive_label = [1]
    elif keyword == 'artist':
        positive_label = [2]
    elif keyword == 'athlete':
        positive_label = [3]
    elif keyword == 'politics':
        positive_label = [4] 
    elif keyword == 'transportation':
        positive_label = [5]
    elif keyword == 'building':
        positive_label = [6]
    elif keyword in ['river','mountain','lake']:
        positive_label = [7]
    elif keyword == 'village':
        positive_label = [8]
    elif keyword == 'animal':
        positive_label = [9]
    elif keyword in ['plant', 'tree']:
        positive_label = [10]
    elif keyword == 'album':
        positive_label = [11]
    elif keyword == 'film':
        positive_label = [12]
    elif keyword in ['novel','publication', 'book']:
        positive_label = [13]   
    else:
        positive_label = [0]
        print('KEYWORD UNKNOWN')

    import os ## TO DO : UNDERSTAND WHY NEEDED HERE
    path = os.getcwd()+'/'+directory+'/'+keyword+str(number_of_loop_over_vocab)+'/'


    # #### IMPORTS AND GLOBAL PARAMETER



    import torch
    from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
    import numpy as np
    import pandas as pd
    import gc
    from joblib import Parallel, delayed
    import random
    from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
    from torch.utils.data import Subset
    import pickle as p
    from tqdm import tqdm
    import os
    from transformers import BertPreTrainedModel, BertModel
    from transformers.modeling_bert import BertOnlyMLMHead
    from torch import nn
    import sys
    from tqdm import tqdm
    from time import time
    import pickle as p
    import matplotlib.pyplot as plt

    from collections import Counter
    from nltk.util import ngrams 
    from itertools import chain
    from nltk.corpus import stopwords
    print('GPU AVAILABLE : ', torch.cuda.is_available())

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # #### Statistics of the Positive Set



    #### DATA FOLDER



    mcp_data = torch.load(path+'mcp_train.pt')
    assumed_label = []
    if len(positive_label) == 1:
        for x in mcp_data['labels']:
            assumed_label.append(positive_label[0])
    else:
        for x in mcp_data['labels']:
            if (x!=-1).sum().item() < 2:
                assumed_label.append(((x!=-1)*x).sum().item())
            elif len(np.unique(x[x!=-1].numpy())) == 1:       
                assumed_label.append(np.unique(x[x!=-1].numpy())[0])
            else:
                assumed_label.append(-1)



    ground = mcp_data['ground_truth'].numpy()
    assumed_label = np.array(assumed_label)
    df = pd.DataFrame([ground, assumed_label]).T
    df.columns = ['ground','assumed']
    df['valid'] = df['ground']==df['assumed']
    precision = df.groupby('assumed')['valid'].sum()/df.groupby('assumed')['assumed'].count()

    mcp_data = torch.load(path+'mcp_train.pt')
    label = torch.LongTensor([1]).repeat(len(mcp_data['labels']))

    number_of_correct_positives = pd.DataFrame(mcp_data['ground_truth'].numpy()).value_counts().values



    # real positive over positive 
    print("PRECISION of the positive set : ", number_of_correct_positives[0]/len(mcp_data['ground_truth']))
    pos_set_accuracy = number_of_correct_positives[0]/len(mcp_data['ground_truth'])
    number_elements_pos_set = len(mcp_data['ground_truth'])
    print("NUMBER OF ELEMENTS in the positive set", number_elements_pos_set)


    # #### DEFINITION OF THE MODEL 



    class LOTClassModel(BertPreTrainedModel):

        def __init__(self, config):
            super().__init__(config)
            self.num_labels = config.num_labels
            self.bert = BertModel(config, add_pooling_layer=False)
            self.cls = BertOnlyMLMHead(config)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.activation = nn.Tanh()
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
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
                trans_states = self.dense(last_hidden_states)
                trans_states = self.activation(trans_states)
                trans_states = self.dropout(trans_states)
                logits = self.classifier(trans_states)
            elif pred_mode == "mlm":
                logits = self.cls(last_hidden_states)
            else:
                sys.exit("Wrong pred_mode!")
            return logits


    # #### DEFINITION OF THE TOKENIZER FOR THE MODEL

    # In[21]:


    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)
    vocab = tokenizer.get_vocab()
    inv_vocab = {k:v for v, k in vocab.items()}


    # #### IMPORTING THE DATA AND PREPROCESSING

    # In[22]:


    data_vocab = torch.load(path+"category_vocab.pt")
    label_data = torch.load(path+"label_name_data.pt")
    train_data = torch.load(path+'train.pt')


    # In[23]:


    ### TEXT AND LABELS LOADING AND FORMATING
    corpus = open(directory+'/'+'train.txt', encoding="utf-8")
    true_labels = open(directory+'/'+'train_labels.txt', encoding="utf-8")
    docs_labels = [doc.strip() for doc in true_labels.readlines()]
    dict_label = {0:[], 1:[], 2:[],3:[]}
    list_label = [int(label) for label in docs_labels]
    for i, label in enumerate(docs_labels):
        dict_label[int(label)].append(i)
    docs = [doc.strip() for doc in corpus.readlines()]


    # In[24]:


    ### ALL KEYWORDS' RELATED WORD ARE STACKED TOGETHER as Tokens (integers)
    category_vocab = []
    for k in data_vocab.keys():
        category_vocab += list(data_vocab[k])


    # In[25]:


    ### CREATION OF THE LIST as Strings
    list_pos_keyword = []
    for w in category_vocab:
        list_pos_keyword.append(inv_vocab[w])


    # #### DEFINITION OF UTILITARY FUNCTIONS

    # In[26]:


    def test(model, number = 1024, test_batch_size = 32,docs = docs, all = False, true_label = positive_label):
        model.eval()
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
        inputs = torch.stack([encode(docs[i])[0].squeeze() for i in test_list])
        attention_mask = torch.stack([encode(docs[i])[1].squeeze() for i in test_list])
        true_labels = torch.stack([torch.tensor(int(list_label[i] in true_label)) for i in test_list])
        test_dataset = TensorDataset(inputs, attention_mask, true_labels)
        test_dataloader = DataLoader(test_dataset, batch_size = test_batch_size)
        with torch.no_grad():
            for batch in test_dataloader:
                inputs_test, attention_test, labels_test = batch
                logits = model(inputs_test.to(device),attention_mask=attention_test.to(device), pred_mode='classification')
                logits_cls = logits[:,0]
                prediction = torch.argmax(logits_cls, -1)
                
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
        if recall+precision > 0:
            f1_score = 2*(recall*precision)/(recall+precision)
            print("F1_score", f1_score)
        else:
            f1_score = None
            print("F1_score Undefined")
        print("Accuracy ", accuracy)
        model.train()
        return accuracy, precision, recall, f1_score



        
        
    def encode(docs, tokenizer = tokenizer, max_length = 200):
        encoded_dict = tokenizer.encode_plus(docs, add_special_tokens=True, max_length=max_length, padding='max_length',
                                                        return_attention_mask=True, truncation=True, return_tensors='pt')
        input_ids = encoded_dict['input_ids']
        attention_masks = encoded_dict['attention_mask']
        return input_ids, attention_masks



    #### THESE ARE ONLY USED WHEN COMPUTING INTERSECTION ON GPUs
    def intersect_tensor(t1, t2, device = 'cuda', mask = None):    
        indices = torch.zeros_like(t1, dtype = torch.uint8, device = device)
        for elem in t2:
            indices = indices | (t1 == elem) 
            indices = indices.to(bool)
            
        if mask is not None:
            indices = indices * mask 
        intersection = t1[indices]  
        return intersection, indices

    def count_similar_words(batch, category_vocab = category_vocab):
        prediction= batch[0]
        input_mask = batch[1]
        masked_pred = prediction[:input_mask.sum().item(),:]
        _ , words = torch.topk(masked_pred, 8, -1)
        counter = 0
        for word in words.squeeze():
            counter += int(len(np.intersect1d(word.numpy(), category_vocab))>0)
            intersect_time = time() - intersect_time_start
            if counter > 0:
                print('break')
                return False
                break
        return True
                
    def occurences(word, vocab = category_vocab):
        return len(np.intersect1d(word.cpu().numpy(), vocab))

    def decode(ids, tokenizer=tokenizer):
        strings = tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return strings


    # #### SELECTION OF TEXTS WITH NO KEYWORDS' RELATED WORDS

    # In[27]:


    negative_doc=[]
    negative_doc_label = []
    for k, doc in tqdm(enumerate(docs)):
        tokenized_doc = tokenizer.tokenize(doc)
        new_doc = []
        wordpcs = []
        label_idx = -1 * torch.ones(512, dtype=torch.long)
        for idx, wordpc in enumerate(tokenized_doc):
            wordpcs.append(wordpc[2:] if wordpc.startswith("##") else wordpc)
            if idx >= 512 - 1: # last index will be [SEP] token
                break
            if idx == len(doc) - 1 or not doc[idx+1].startswith("##"):
                word = ''.join(wordpcs)
                if word in list_pos_keyword:
                    label_idx[idx] = 0
                    break
                new_word = ''.join(wordpcs)
                if new_word != tokenizer.unk_token:
                    idx += len(wordpcs)
                    new_doc.append(new_word)
                wordpcs = []
        if (label_idx>=0).any():
            continue
        else:
            negative_doc_label.append(list_label[k])
            negative_doc.append(doc)
        


    # #### Metrics of the Negative Set before the use of any language model

    # In[28]:


    print("Negative pre-set", len(negative_doc))
    print("Precision pre-set, ", len([k for k in negative_doc_label if k not in positive_label])/len(negative_doc_label))


    # #### FORMATING THE NEGATIVE SET - ENCODING AND FORMATING INTO TENSORS

    # In[29]:


    inputs_list = []
    masks_list = []
    for doc in tqdm(negative_doc):
        input_ids, input_mask = encode(doc)
        inputs_list.append(input_ids)
        masks_list.append(input_mask)
    input_tensor = torch.stack(inputs_list).squeeze()
    mask_tensor = torch.stack(masks_list).squeeze()
    label_tensor = torch.stack([torch.tensor(i).unsqueeze(0) for i in negative_doc_label])
    dataset = torch.utils.data.TensorDataset(input_tensor,mask_tensor, label_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle = False, batch_size = 8)


    # #### CREATION OF MODEL AND HYPERPARAMETERS FOR THE FILTERING

    # In[30]:


    model = LOTClassModel.from_pretrained('bert-large-uncased',
                                            output_attentions=False,
                                            output_hidden_states=False,
                                            num_labels=2).to('cuda')
    # Parameters controlling the size of the neg set
    relative_factor_pos_neg = 3
    early_stopping = True

    verified_negative = []
    correct_label = 0
    verbose = True
    topk = 15
    vocab = torch.tensor(category_vocab).to(device)
    min_similar_words = 0
    max_category_word = 0


    # #### COMPUTING THE NEGATIVE SET BASED ON THE PRETRAINED LANGUAGE MODEL

    # In[31]:


    with torch.no_grad():
        for k, batch in tqdm(enumerate(dataloader)):

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
                counter = 0
                # Loop over the words in each document
                for word in words.squeeze():
                    counter += int(len(np.intersect1d(word.cpu().numpy(), category_vocab))>min_similar_words)
                    if counter > max_category_word:
                        break
                if counter <= max_category_word:             
                    verified_negative.append(k*4+i)
                    if label_id[i] not in positive_label:
                        correct_label += 1 
            if (len(verified_negative) > relative_factor_pos_neg * number_elements_pos_set) and early_stopping:
                break
            if k%100 == 0 and verbose:
                if len(verified_negative)>0:
                    print('accuracy :', correct_label/len(verified_negative))
                    print('number of elements retrieved', len(verified_negative))


    neg_set_accuracy = correct_label/len(verified_negative)    
            
            
        


    # #### EXPORT THE SET AND THE DATALOADER

    # In[32]:


    p.dump(verified_negative, open(path+'verified_negative.p','wb'))
    p.dump(dataloader, open(path+'dataloader.p','wb'))


    # # II/ TRAINING SECTION

    # #### LOADING OF THE SETS AND FORMATING

    # In[33]:


    # Negative Set
    new_verified_negative = p.load(open(path+'verified_negative.p','rb'))
    new_dataloader = p.load(open(path+'dataloader.p','rb'))

    # Positive Set
    mcp_data = torch.load(path+'mcp_train.pt')
    label = torch.LongTensor([1]).repeat(len(mcp_data['labels']))

    # Formating
    negative_dataset = Subset(new_dataloader.dataset, new_verified_negative)
    positive_dataset = torch.utils.data.TensorDataset(mcp_data['input_ids'], mcp_data['attention_masks'], mcp_data['labels'])

    ## TO DO : Words statistics on both sets#



    # #### CONSTRUCTION OF THE DATASETS AND OF THE WEIGHTED SAMPLER

    # In[35]:


    ####### Construction of the weighted sampler based on sets' sizes and of the target vector #########

    target = np.hstack((np.zeros(int(len(negative_dataset)), dtype=np.int32),
                        np.ones(int(len(positive_dataset)), dtype=np.int32)))

    class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])

    samples_weight = torch.from_numpy(samples_weight)
    samples_weigth = samples_weight.double()


    target = torch.from_numpy(target).long()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))


    # #### DEFINTION OF THE BATCH SIZE
    # 

    # In[36]:


    batch_size = 16


    # #### CONSTRUCTION OF THE DATALOADER

    # In[37]:


    data = torch.stack([negative_data[0][:200] for negative_data in negative_dataset] + 
                [positive_data[0][:200] for positive_data in positive_dataset])

    mask = torch.stack([negative_data[1][:200] for negative_data in negative_dataset] + 
                [positive_data[1][:200] for positive_data in positive_dataset])

    train_dataset = torch.utils.data.TensorDataset(data,mask, target)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, sampler=sampler)


    # #### MODEL INSTANTIATION

    # In[38]:


    model = LOTClassModel.from_pretrained('bert-large-uncased',
                                            output_attentions=False,
                                            output_hidden_states=False,
                                            num_labels=2).to(device)


    # #### TRAINING HYPERPARAMETERS AND PARAMETERS

    # In[39]:


    accum_steps = 8
    epochs = 3
    learning_rate = 1e-5
    train_loss = nn.CrossEntropyLoss()
    total_steps = len(train_loader) * epochs / accum_steps
    number_of_mask = 2

    parameters = {'epochs':epochs, 'learning_rate':learning_rate, 'number_of_mask':number_of_mask,
                'accum_steps':accum_steps, 
                'batch_size': batch_size, 
                'pos_set' : len(positive_dataset),
                'pos_set_accuracy' : pos_set_accuracy,
                'neg_set' : len(negative_dataset),
                'neg_set_accuracy' :neg_set_accuracy,
                'loop_over_vocab' : number_of_loop_over_vocab,
                'keyword' : keyword}


    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*total_steps, num_training_steps=total_steps)

    number_of_mask = 1 
    # Metrics
    losses_track = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    global_steps = []


    # #### TRAINING LOOP

    # In[40]:


    model.train()
    try:
        for i in range(epochs):
            model.train()
            total_train_loss = 0
            model.zero_grad()
            print('Epoch : ', i)
            for j, batch in enumerate(train_loader):
                input_ids = batch[0].to(device)
                input_mask = batch[1].to(device)
                labels = batch[2].to(device)


                ### RANDOM MASKING
                random_masking = random.choices(list(range(199)),k = number_of_mask * input_ids.size(0))
                for i, mask_pos in enumerate(random_masking):
                    input_ids[i%input_ids.size(0),mask_pos+1] = tokenizer.get_vocab()[tokenizer.mask_token]
                
                ### PREDICTION
                logits = model(input_ids, 
                            pred_mode="classification",
                            token_type_ids=None, 
                            attention_mask=input_mask)
                ### LOSS
                logits_cls = logits[:,0]
                loss = train_loss(logits_cls.view(-1, 2), labels.view(-1)) / accum_steps            
                total_train_loss += loss.item()
                loss.backward()
                if (j+1) % accum_steps == 0:
                    # Clip the norm of the gradients to 1.0.
                    
                    losses_track.append(loss*accum_steps)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    
                if j % (3*accum_steps) == 0 :
                    print('loss',loss*accum_steps)
                    accuracy, precision, recall, f1_score = test(model, number = 1024)
                    losses_track.append(loss*accum_steps)
                    accuracies.append(accuracy)
                    precisions.append(precision)
                    recalls.append(recall)
                    f1_scores.append(f1_score)
                    global_steps.append(j)
            avg_train_loss = torch.tensor([total_train_loss / len(train_loader) * accum_steps]).to(device)
            print(f"Average training loss: {avg_train_loss.mean().item()}")

    except RuntimeError as err:
        print(err)


    # In[41]:



    plt.plot(f1_scores)


    # #### GLOBAL TEST

    # In[42]:


    final_results = {}
    res = test(model = model, all = True)
    final_results['accuracy'] =  res[0]
    final_results['precision'] = res[1]
    final_results['recall'] = res[2]
    final_results['f1_score'] = res[3]


    # #### SAVE EVERYTHING

    # In[43]:


    p.dump(accuracies,open(path+'accuracy.p','wb'))
    p.dump(losses_track,open(path+'loss.p','wb'))
    p.dump(precisions,open(path+'precision.p','wb'))
    p.dump(recalls,open(path+'recall.p','wb'))
    p.dump(f1_scores,open(path+'f1_score.p','wb'))
    p.dump(parameters, open(path+'parameters.p', 'wb'))


    # In[44]:


    p.dump(final_results, open(path+'final_results.p', 'wb'))


    # #### SAVE MODEL

    # In[45]:


    torch.save(model.state_dict(), path+'model.pt')


    # In[ ]:
if __name__ == "__main__":
    main()



