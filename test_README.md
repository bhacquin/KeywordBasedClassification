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

## Metrics - True Labels
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

The labels to compute the metrics are expected to be in a separate text file with one label per line. The labels file and text file are expected to show texts in the same order.

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

## Reproducing the Results - Short

## Reproducing the Results - Detailed

### Environment
#### Dataset
#### Keywords
#### Hyperparameters
#### Command line
``` python
trainer = ClassifTrainer()
```
### 'Category Vocabulary per Keyword'
```python
trainer.category_vocab()
```
## Running on new datasets

## Next Steps - Improvements
Uncertainty use.
## Citations
