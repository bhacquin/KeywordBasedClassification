#!/usr/bin/env python
# coding: utf-8

# In[5]:


import torch
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import numpy as np
import gc


# In[6]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
vocab = tokenizer.get_vocab()
inv_vocab = {k:v for v, k in vocab.items()}


# In[7]:


data_vocab = torch.load("category_vocab.pt")


# In[9]:


data = torch.load("label_name_data.pt")


# In[10]:


train_data = torch.load('train.pt')


# In[12]:


import os
corpus = open('train.txt', encoding="utf-8")
labels = open('train_labels.txt', encoding="utf-8")
docs_labels = [doc.strip() for doc in labels.readlines()]
dict_label = {1:[], 2:[],3:[],0:[]}
list_label = [int(label) for label in docs_labels]
for i, label in enumerate(docs_labels):
    dict_label[int(label)].append(i)
docs = [doc.strip() for doc in corpus.readlines()]


# In[14]:


# Creer la liste de mots positifs
category_vocab = list(data_vocab[0])+list(data_vocab[1])+list(data_vocab[2])
list_pos_keyword = []
for w in category_vocab:
    list_pos_keyword.append(inv_vocab[w])


# In[15]:


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
    


# In[16]:


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
                                           num_labels=5).to('cuda')


# In[17]:


inputs_list = []
masks_list = []
for doc in tqdm(negative_doc):
    input_ids, input_mask = encode(doc)
    inputs_list.append(input_ids)
    masks_list.append(input_mask)


# In[18]:


input_tensor = torch.stack(inputs_list).squeeze()
mask_tensor = torch.stack(masks_list).squeeze()
dataset = torch.utils.data.TensorDataset(input_tensor,mask_tensor)
dataloader = torch.utils.data.DataLoader(dataset, shuffle = False, batch_size = 4)


# In[ ]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
from time import time
verified_negative = []
with torch.no_grad():
    for k, batch in tqdm(enumerate(dataloader)):
        start_time = time()
        input_ids, input_mask = batch
        predictions = model(input_ids.to(device),
                        pred_mode="mlm",
                        token_type_ids=None, 
                        attention_mask=input_mask.to(device))
        end_prediction_time = time()
#         print('Prediction time', end_prediction_time-start_time)
        for i, doc in enumerate(predictions.cpu()):
            masked_pred = doc[:input_mask[i].sum().item(),:]
            _ , words = torch.topk(masked_pred, 6, -1)
            counter = 0
            for word in words.squeeze():
                counter += int(len(np.intersect1d(word.numpy(), category_vocab))>0)
                if counter > 0:
                    break
            if counter <= 0 :
                verified_negative.append(k*4+i)
#         print('Loop time', time()-end_prediction_time)
        del predictions
        gc.collect()
        torch.cuda.empty_cache()
        
        
    


# In[14]:


np.intersect1d(words.numpy().flatten(),category_vocab)


# In[23]:


len(verified_negative)


# In[ ]:


import pickle as p
p.dump(verified_negative, open('verified_negative.p',wb))
p.dump(dataloader, open('dataloa.p',wb))

