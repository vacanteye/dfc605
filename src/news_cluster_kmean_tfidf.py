import sqlite3, pdb
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances

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

# Configurations
max_iter = 300
n_data_size = 100
n_clusters = 5
n_cluster_freqwords = 7
ngram_range = (1,2)
    
# Load Data
conn = sqlite3.connect('../dat/news_daily_20200501.db')
df = pd.read_sql_query('SELECT date,title,nouns FROM news_table LIMIT {}'.format(n_data_size), conn)
conn.close()

titles = df['title'].tolist()
docs = df['nouns'].tolist()
#docs = df['title'].tolist()

# Vectorizing
vectorizer = TfidfVectorizer(
    ngram_range=ngram_range
)

# Document-Term Matrix
dtm = vectorizer.fit_transform(docs)

# Features
words = vectorizer.get_feature_names()

# L2 Normalizing
#X = normalize(dtm)
X = dtm

# Cluster with K-Means
kmeans = KMeans(
    n_clusters=n_clusters,
    max_iter = max_iter,
    verbose = 0
).fit(X)

# Clustring Results
clusters = kmeans.labels_
centers = kmeans.cluster_centers_

# Insert Result Column
df['cluster'] = clusters
df['distance'] = 0

for i in range(n_clusters):

    #Select row with given cluster
    cdf = df.loc[df['cluster'] == i].sort_index(ascending=False)  

    #Calculate frequent words : sum of rows to vector
    count_vector = dtm[cdf.index].sum(axis=0).getA().ravel()

    #Calculate Distance
    center = sparse.csr_matrix(centers[i])
    dist_vectors = pairwise_distances(X[cdf.index], center, metric='euclidean').ravel()
    cdf['distance'] = dist_vectors

    cluster_topwords = []
    if count_vector is not None:
        word_indexes = count_vector.argsort()[::-1][:n_cluster_freqwords] #top words
        for word_index in word_indexes:
            keyword = words[word_index]
            count = count_vector[word_index]
            cluster_topwords.append('{}({:.2f})'.format(keyword, count))

    cluster_topwords = ','.join(cluster_topwords)

    # Cluster Infos
    print("Cluster:{} Articles:{} Topwords:{}".format(i+1, len(cdf), cluster_topwords))

    cluster_titles = ['{:.2f} - {} - {}'.format(row['distance'], row['date'], row['title']) for i, row in cdf.iterrows()]
    cluster_titles = '\n '.join(cluster_titles)
    print(' {}'.format(cluster_titles))
    print()

print('title:{}, cluster:{}'.format(len(titles), n_clusters))
