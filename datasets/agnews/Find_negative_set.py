#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import numpy as np
import gc
from joblib import Parallel, delayed
positive_label = [0,2,3]


# In[2]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
vocab = tokenizer.get_vocab()
inv_vocab = {k:v for v, k in vocab.items()}


# In[3]:


data_vocab = torch.load("category_vocab.pt")


# In[4]:


data = torch.load("label_name_data.pt")


# In[5]:


train_data = torch.load('train.pt')


# In[6]:


import os
corpus = open('train.txt', encoding="utf-8")
labels = open('train_labels.txt', encoding="utf-8")
docs_labels = [doc.strip() for doc in labels.readlines()]
dict_label = {0:[], 1:[], 2:[],3:[]}
list_label = [int(label) for label in docs_labels]
for i, label in enumerate(docs_labels):
    dict_label[int(label)].append(i)
docs = [doc.strip() for doc in corpus.readlines()]


# In[7]:


category_vocab = []
for k in data_vocab.keys():
    category_vocab += list(data_vocab[k])


# In[8]:


# Creer la liste de mots positifs
# category_vocab = list(data_vocab[0])+list(data_vocab[1])+list(data_vocab[2])
list_pos_keyword = []
for w in category_vocab:
    list_pos_keyword.append(inv_vocab[w])


# In[9]:


from tqdm import tqdm
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
                # replace label names that are not in tokenizer's vocabulary with the [MASK] token
    #             if word not in vocab:
    #                 wordpcs = [tokenizer.mask_token]
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
    


# In[11]:


print("Negative pre-set", len(negative_doc))
print("Accuracy pre-set, ", len([k for k in negative_doc_label if k not in positive_label])/len(negative_doc_label))


# In[12]:


from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_bert import BertOnlyMLMHead
from torch import nn
import sys



def encode(docs, tokenizer = tokenizer):
    encoded_dict = tokenizer.encode_plus(docs, add_special_tokens=True, max_length=512, padding='max_length',
                                                    return_attention_mask=True, truncation=True, return_tensors='pt')
    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    return input_ids, attention_masks


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
    
    
model = LOTClassModel.from_pretrained('bert-base-uncased',
                                           output_attentions=False,
                                           output_hidden_states=False,
                                           num_labels=2).to('cuda')


# In[13]:


inputs_list = []
masks_list = []
for doc in tqdm(negative_doc):
    input_ids, input_mask = encode(doc)
    inputs_list.append(input_ids)
    masks_list.append(input_mask)


# In[14]:


input_tensor = torch.stack(inputs_list).squeeze()
mask_tensor = torch.stack(masks_list).squeeze()
label_tensor = torch.stack([torch.tensor(i).unsqueeze(0) for i in negative_doc_label])
dataset = torch.utils.data.TensorDataset(input_tensor,mask_tensor, label_tensor)
dataloader = torch.utils.data.DataLoader(dataset, shuffle = False, batch_size = 8)


# In[15]:


def intersect_tensor(t1, t2, device = 'cuda', mask = None):    
    indices = torch.zeros_like(t1, dtype = torch.uint8, device = device)
    for elem in t2:
        indices = indices | (t1 == elem) 
        indices = indices.to(bool)
        
    if mask is not None:
        indices = indices * mask 
    intersection = t1[indices]  
    return intersection, indices


# In[16]:


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


# In[27]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
from time import time
verified_negative = []
correct_label = 0
verbose = False
topk = 30
vocab = torch.tensor(category_vocab).to(device)
min_similar_words = 1
num_cpus = 8
with torch.no_grad():
    for k, batch in tqdm(enumerate(dataloader)):
        start_time = time()
        input_ids, input_mask, label_id = batch
        predictions = model(input_ids.to(device),
                        pred_mode="mlm",
                        token_type_ids=None, 
                        attention_mask=input_mask.to(device))
        end_prediction_time = time()
        
        
    ########### GPU    
#         intersection, indices = intersect_tensor(torch.topk(predictions,topk,-1)[1],vocab, 
#                                                 mask = input_mask.unsqueeze(2).repeat(1,1,topk).to(device))
#         counts_similar_word_per_word = indices.sum(-1) 
        
#         counts = (counts_similar_word_per_word>min_similar_words).sum(-1)
#         end_intersection_time = time()
        
#         indices_count = torch.where(counts<=0)[0]
#         for j in indices_count:
#             i = j.item()

    ################## CPU ####################""
#         counts = Parallel(n_jobs=num_cpus)(delayed(count_similar_words)(batch) for batch in zip(predictions.cpu(), input_mask))
    
    
    
        for i, doc in enumerate(predictions.cpu()):
            start_loop = time()
            masked_pred = doc[:input_mask[i].sum().item(),:]
            _ , words = torch.topk(masked_pred, topk, -1)
            counter = 0
            topk_time = time()
            
#             counts = Parallel(n_jobs=num_cpus)(delayed(occurences)(word) 
#                                                    for word in words.squeeze())
#             counter += len(np.where(np.array(counts)>min_similar_words)[0])
            for word in words.squeeze():
#                 counter += int(len(intersect_tensor(word, vocab))>0)
                counter += int(len(np.intersect1d(word.cpu().numpy(), category_vocab))>0)
                intersect_time = time() - intersect_time_start
                if counter > 0:
                    
                    break

#             j = i.item()
#         for i in np.where(np.array(counts))[0]:

            if counter <= 0:             
                verified_negative.append(k*4+i)
                if label_id[i] not in positive_label:
                    correct_label += 1 

        end_counting_time = time()    
        if k%100 == 0:
            if len(verified_negative)>0:
                print('accuracy :', correct_label/len(verified_negative))
                print('number of elements retrieved', len(verified_negative))
#         if verbose:
#             print('Prediction time', end_prediction_time-start_time) 
# #             print('bascule cpu', start_loop-end_prediction_time)
#             print('Intersection time', end_intersection_time-end_prediction_time)
# #             print('topk time', topk_time-start_loop)
#             print('counting time', end_counting_time-end_intersection_time)
        del predictions
        gc.collect()
        torch.cuda.empty_cache()
        
        
        
    


# In[24]:


del predictions
gc.collect()
torch.cuda.empty_cache()
        


# In[ ]:


solutions = [k for k in verified_negative if negative_doc_label[k]!=1]


# In[ ]:


print(len(np.intersect1d(solutions,verified_negative))/len(verified_negative))


# In[ ]:


import pickle as p
p.dump(verified_negative, open('verified_negative_sports.p','wb'))
p.dump(dataloader, open('dataloader_sports.p','wb'))


# In[ ]:


import pickle as p
new_verified_negative = p.load(open('verified_negative_sports.p','rb'))
new_dataloader = p.load(open('dataloader_sports.p','rb'))


# In[ ]:


# import pickle as p
# p.dump(verified_negative, open('verified_negative.p','wb'))
# p.dump(dataloader, open('dataloader.p','wb'))


# ### top k = top 8 -> ~0.8 accuracy, 400 datapoints
