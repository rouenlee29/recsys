"""
Identifies top k number of keywords in research papers. The keywords are chosen based on tfidf values.
Adapted from Karpathy's arxiv recommender system: https://github.com/karpathy/arxiv-sanity-preserver 
"""
import os
import pickle
from random import shuffle, seed

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import Config, safe_pickle_dump


seed(1337)
max_train = 5000 # max number of tfidf training documents (chosen randomly), for memory efficiency
max_features = 5000


"""
check on a smaller sample size 
"""
import glob

def listdir_nohidden(path):
   return glob.glob(os.path.join(path, '*'))
    
my_path = os.getcwd() 
name = listdir_nohidden(my_path + '/data/txt')

txt_paths = [s.replace(my_path + '/', '') for s in name]
pids = [s.replace('data/txt/', '') for s in txt_paths]
pids = [s.replace('.pdf.txt', '') for s in pids]

"""
Lemmatise tokens using spacy. 
Ignore spacess, and only take tokens that have more than 1 character, and is a verb or noun 
"""    
import spacy

class LemmaTokenizer(object):
    def __init__(self):
        self.spacynlp = spacy.load('en')
    def __call__(self, doc):
        nlpdoc = self.spacynlp(doc)
        nlpdoc = [token.lemma_ for token in nlpdoc if (len(token.lemma_) > 1) and (token.lemma_.isspace() == 0) and ((token.pos_ == 'NOUN') or (token.pos_ == 'VERB'))]
        return nlpdoc

# compute tfidf vectors with scikit
v = TfidfVectorizer(input='content', 
        encoding='utf-8', decode_error='replace', strip_accents='unicode', 
        lowercase=True, tokenizer=LemmaTokenizer(), 
        analyzer='word', stop_words='english', 
        token_pattern=r'(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b',
        ngram_range=(1, 1), max_features = max_features, 
        norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True,
        max_df=1.0, min_df=1)

# create an iterator object to conserve memory
def make_corpus(paths):
  for p in paths:
    with open(p, 'r') as f:
      txt = f.read()
    yield txt

# train
train_txt_paths = list(txt_paths) # duplicate
shuffle(train_txt_paths) # shuffle
train_txt_paths = train_txt_paths[:min(len(train_txt_paths), max_train)] # crop
print("training on %d documents..." % (len(train_txt_paths), ))
train_corpus = make_corpus(train_txt_paths)
X = v.fit_transform(train_corpus)
vocab = v.get_feature_names()

"""
get top most important words for each doc in a dictionary
"""
keywords = {}
top = 3 #top 3 words

for i in range(len(train_txt_paths)):
    
  row = X.getrow(i).toarray()[0].ravel()
  top_indicies = row.argsort()[-top:]
  #top_ten_values = row[row.argsort()[-10:]]
  
  mylist = []
  for j in range(len(top_indicies)):  
    mylist.append(vocab[top_indicies[j]])

  keywords[train_txt_paths[i]] = mylist
  
print(keywords)
