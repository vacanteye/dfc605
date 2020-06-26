import sqlite3, pdb
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import NMF, LatentDirichletAllocation

from scipy import sparse
from scipy.spatial import distance

from konlpy.tag import Hannanum

tokenizer = Hannanum()
stopwords = []

def preprocessing(doc):
    nouns = tokenizer.nouns(doc)
    return ' '.join(nouns)

def read_dic(fname):
    results = []
    fp = open(fname, 'r')
    for line in fp:
        results.append(line.splitlines()[0])
    fp.close()
    return results

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #{}: ".format(topic_idx+1)
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

# Configurations
max_iter = 10
n_samples = 2000
n_topics = 50
n_top_words = 7
ngram_range = (1,2)
    
# Load Data
conn = sqlite3.connect('../dat/news_daily_20200501.db')
df = pd.read_sql_query('SELECT date,title,nouns FROM news_table LIMIT {}'.format(n_samples), conn)
conn.close()

titles = df['title'].tolist()
docs = df['nouns'].tolist()
#docs = df['title'].tolist()

# Vectorizing
vectorizer = CountVectorizer(
    ngram_range=ngram_range
)

# Document-Term Matrix
dtm = vectorizer.fit_transform(docs)

# Normalize : None
X = dtm

# Features
words = vectorizer.get_feature_names()

# Cluster with LDA
lda = LatentDirichletAllocation(
    n_components=n_topics, max_iter=max_iter,
    learning_method='online',
    learning_offset=50.,
    random_state=0,
    verbose=1
)

lda.fit(X)

tf_feature_names = vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)


