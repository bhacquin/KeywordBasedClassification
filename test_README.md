# Keyword Based Classification - Keyword based research

The code and the methods are widely inspired by **Text Classification Using Label Names Only: A Language Model Self-Training Approach**, published in EMNLP 2020.

## Requirements 

At least one GPU is needed to run the code and it needs enough memory to run the chosen Language Model.

Python 3.8 has been used, and Python 3.6 or above is strongly recommended because of possible package incompatibility issues with earlier versions.

To install the required packages, please type the following command:

```bash

pip install -r requirements.txt

```
## Method - Abstract
The model learns to retrieve texts of potential interests for a specific user using keywords only. 

### Improvements over LOTClass algorithm

Our model learns to classify texts only based upon a number of keywords provided by the user.

* In opposition to what was done in LOTClass, our **model does not require the keywords to be the underlying labels of the dataset**, nor do the keywords need to represent the whole dataset faithfully.
* The idea is that a user can easily describe his/her interests but cannot easily know what the dataset consists of , nor could he/she describe what he/she is *not* interested in. Hence **our algorithm can work without any negative keywords**.
* Our model can work with **as few as one single keyword**

### First Stage - Category Vocabulary
Based on **Text Classification Using Label Names Only: A Language Model Self-Training Approach**, we follow the same approach and will create sets of closely related words around each keyword provided by the user.

### Second Stage - Weak Supervision
Based on the categories' vocabularies and the method described in **Text Classification Using Label Names Only: A Language Model Self-Training Approach**, the model label automatically a number of text that are very likely to be related to one specific keyword. 

Here we have also enriched the method of the paper by automatically digging other keyword if the one given by the user does not lead to stable category vocabulary.

Once that has been done, the model will label automatically texts that are far from all those categories as negative or class of no-interest. 

With those automatically found labels, we fine-tuned a language model on them.

### Third Stage - Self Training
As described in **Text Classification Using Label Names Only: A Language Model Self-Training Approach**, the model is then self trained to boost its confidence on the whole dataset this time.
Self-Training has been seen to improve significantly the recall without really deteriorating the precision, hence self-training results in a much higher F1-score.



### Metrics - True Labels
The model does not use the true labels of the dataset at any point. Those labels are only use to track metrics and evaluate the performance of the model. We study the classic following metrics:
* F1-score
* Recall
* Precision
* Accuracy

## Datasets
We have been working on four datasets : 
* AGNews, 120k english news labelled as : Sport, Politics, Business or Technology (MultiClass)
* DBPedia datasets : a large dataset with different granurality of labels.
* IMDB : 50k Movie reviews labelled as positive or negative
* Amazon : ?

In the datasets folder, there are four ```get_data.sh``` for downloading the datasets we have been using. As the scripts might not be updated anymore, those datasets can also be found online. 

### Format
The texts are supposed to be all in one text file with one text per line.

```
text1
text2
text3
...
```

The labels to compute the metrics are expected to be in a separate text file with one label per line. The labels file and text file are expected to show texts in the same order.

```
LabelOFText1
LabelOfText2
LabelOfText3
...
```

The keywords are supposed to be put in a different file under this format:
```
keyword1;positive 
keyword2;negative
keyword3;negative
keyword4;positive
....
```
Keywords are expected to be different and not synonyme if you intend to put synonym please put it this:

```
keyword1,synonym11;positive
keyword2,synonyme21,synonym22;negative
....
```

## Results
Results are obviously depending on what the user looks for. 
Here we take as an hypothesis that one is looking for a combination of the known labels so we can track results with proper metrics.

### AGNews

Keywords      | F1 - score    |    Precision  |     Recall    |  
------------- | ------------- |-------------- | ------------  |
Sport         | 85            | ?             | ??            |
Politics      | 75            | 66.5             | 88.5            |
{Business, Technology}  | 92%  |  ??          | ??            |
{Politics, Business, Technology}  |   90.7%   |  98.6%     |    84%     |

To compare with other benchmark
Model         | F1 - score    |    Precision  |     Recall    |  
------------- | ------------- |-------------- | ------------  |
Cate          | ??            | ??            |     ??        |

### DBPedia

Keywords      | F1 - score    |    Precision  |     Recall    |  
------------- | ------------- |-------------- | ------------  |
athlete         | ??            | ?             | ??            |
artist          | ??            | ?             | ??            |
{tree, river}   |    ??        |    ??          | ??            |
{river, mountain, hill}  |   ??   |  ??     |    ??     |


### IMDB

Keywords      | F1 - score    |    Precision  |     Recall    |  
------------- | ------------- |-------------- | ------------  |
Good          | 80.5            | 77.0%            | 84.2%            |
Bad           | ??            | ??            | ??            |

To compare with other benchmark
Model         | F1 - score    |    Precision  |     Recall    |  
------------- | ------------- |-------------- | ------------  |
Cate          | ??            | ?             | ??            |



## Reproducing the Results - Short

### Data
The data must be in the right format and all files in the same directory.
Add the keywords as explained in a text file in the same directory.

It is better to split your dataset into train and test files, both with the same format

Assuming you gave keywords and expect to retrieve texts with label 1 and 5 for instance.
Then simply run from the root directory:
```bash
python train.py --dataset_dir your_data_directory --keywords_file file_name --train_file file_name --test_file file_name --train_label_file file_name --test_label_file file_name -- true_label 1 5
```
with the proper parameters

## Reproducing the Results - Detailed
Here will be described in more detailed how to run things separetely. 
For this, I will explain the ```train.py``` step by step.
 
#### Hyperparameters
The ```train.py``` has a number of hyperparameters that can be tuned. There are described on the file.
#### Step by Step
##### Instanciation
``` python
trainer = ClassifTrainer(args)
```
This line instantiates the trainer with all the parameters provided or the default ones.
At this point, the keywords are read, their numbers of occurences in the dataset have been tracked. If keywords are adjectives they have also potentially been increased to their antonyms if those where not provided as other keywords.
The data has also already been encoded using the Language Model Tokenizer into pytorch tensors.

#### 'Category Vocabulary per Keyword'
```python
trainer.category_vocab(top_pred_num=args.top_pred_num, match_threshold=args.match_threshold, loader_name="mcp_train.pt")
```
This method creates one category vocabulary per keyword in order to use it later. So it takes the set of keywords defined at instanciation and build a set of similar words (context-based), namely words that are likely to replace the keywords accross the dataset. 

In case on of the keyword for instance is hardly present in the dataset, this set of similar words will be refined by running the algorithm on the most similar word to the keyword as a refinement step.
Say, 'tech' is the keyword, in the set of similar words 'technology' comes first, then the set will be refined using 'technology' as a keyword


#### Weak Automatic Labelling using LMs
```python
trainer.prepare_mcp(top_pred_num=args.top_pred_num, match_threshold=args.match_threshold, loader_name="mcp_train.pt")
```
This method will automatically labels a few of the texts in the dataset based on the category vocabulary built before and on the Language Model chosen. 

Subsets of the dataset are constructed out of the texts the most likely to be related to the keywords provided, likelihood being assesed with the category built before.

As a result, the method builds a subset of the full dataset with noisy labels.

#### Statistics on the precision of those subsets
```python
trainer.training_set_statistics()
```
This methods analyses the previously built subset of the dataset and its noisy labels. It computes the accuracy of the supposed positive sets, and supposed negative sets regarding the ground true labels.


#### Computes set of negative examples
```python
trainer.compute_preset_negative()
```
This returns a set of texts with no occurence of any word related to a positive keyword.

```python
trainer.compute_set_negative()
```
Out of the set of likely negative texts just built before, this method selects the most likely ones to be negative using the LM.
The idea is to retrieve texts that neither the positive keywords nor the negative keywords may cover. 
In case the number of texts thus retrieved is too small, then this is simply dropped.

#### Training
```python
trainer.train()
```
Train the models on the noisy sets builts before using Gradient Descent.

#### Self Training
```python
trainer.self_train(epochs=args.self_train_epochs, loader_name=args.final_model)
```
The method refines the model on the whole dataset this time by using the most 'confident' predictions of the trained model (strecthing predictions either to 1 or 0).

## Running on new datasets

## Next Steps - Improvements
Uncertainty use.



## References

* Jacob Devlin, Ming-Wei Chang, Kenton Lee, andKristina Toutanova. 2019. Bert: Pre-training of deepbidirectional transformers for language understand-ing. InNAACL-HLT
* Yu Meng, Jiaming Shen, Chao Zhang, and Jiawei Han.2018. Weakly-supervised neural text classification.InCIKM
* Yu Meng, Jiaxin Huang, Guangyuan Wang, ZihanWang, Chao Zhang, Yu Zhang, and Jiawei Han.2020a. Discriminative topic mining via category-name guided text embedding. InWWW
* Yu Meng, Yunyi Zhang, Jiaxin Huang, Chenyan Xiong, Heng Ji, CHao Zhang, Jiawei Han. 2020. Text Classification Using Label Names Only: A Language Model Self-Training Approach.EMMNLP

