"""
Reads txt files of all papers and computes tfidf vectors for all papers.
Dumps results to file tfidf.p
"""
import os
import pickle
from random import shuffle, seed
import glob
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# james edit
from pprint import pprint

from utils import Config, safe_pickle_dump

seed(1337)
max_train = 5000 # max number of tfidf training documents (chosen randomly), for memory efficiency
max_features = 5000

# read database
# db = pickle.load(open(Config.db_path, 'rb'))

# txt_paths and glob
def listdir_nohidden(path):
   return glob.glob(os.path.join(path, '*'))
    
my_path = os.getcwd() 
name = listdir_nohidden(my_path + '/data/txt')

txt_paths = [s.replace('/Users/rouenlee/recsys/ico/', '') for s in name]
pids = [s.replace('data/txt/', '') for s in txt_paths]
pids = [s.replace('.pdf.txt', '') for s in pids]



# n = 0
# for pid,j in db.items():
#   n += 1
#   idvv = '%sv%d' % (j['_rawid'], j['_version'])
#   txt_path = os.path.join('data', 'txt', idvv) + '.pdf.txt'
#   if os.path.isfile(txt_path): # some pdfs dont translate to txt
#     with open(txt_path, 'r') as f:
#       txt = f.read()
#     if len(txt) > 1000 and len(txt) < 500000: # 500K is VERY conservative upper bound
#       txt_paths.append(txt_path) # todo later: maybe filter or something some of them
#       pids.append(idvv)
#       print("read %d/%d (%s) with %d chars" % (n, len(db), idvv, len(txt)))
#     else:
#       print("skipped %d/%d (%s) with %d chars: suspicious!" % (n, len(db), idvv, len(txt)))
#   else:
#     print("could not find %s in txt folder." % (txt_path, ))
# print("in total read in %d text files out of %d db entries." % (len(txt_paths), len(db)))

# compute tfidf vectors with scikits
v = TfidfVectorizer(input='content', 
        encoding='utf-8', decode_error='replace', strip_accents='unicode', 
        lowercase=True, analyzer='word', stop_words='english', 
        token_pattern=r'(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b',
        ngram_range=(1, 2), max_features = max_features, 
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
# print(X)

# jedit
print(list(train_corpus))


# print( X.shape )  #(2, 5000)
# print(type(X))

row = X.getrow(0).toarray()[0].ravel()
top_ten_indicies = row.argsort()[-10:]
#top_ten_values = row[row.argsort()[-10:]]
print(top_ten_indicies)
#print(top_ten_values)

#mafkjds;fhl;k 


# https://stackoverflow.com/questions/31790819/scipy-sparse-csr-matrix-how-to-get-top-ten-values-and-indices



