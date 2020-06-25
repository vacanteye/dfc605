import sqlite3, pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import AgglomerativeClustering

from scipy.cluster.hierarchy import dendrogram
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

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

# Configurations
n_data_size = 100
n_clusters = 5
n_cluster_freqwords = 7
    
# Load Data
conn = sqlite3.connect('../dat/news_daily_20200501.db')
df = pd.read_sql_query('SELECT title,nouns FROM news_table LIMIT {}'.format(n_data_size), conn)
conn.close()

titles = df['title'].tolist()
docs = df['nouns'].tolist()
#docs = df['title'].tolist()

# Vectorizing
vectorizer = CountVectorizer()

# Document-Term Matrix
dtm = vectorizer.fit_transform(docs)

# Features
words = vectorizer.get_feature_names()

# L2 Normalizing
X = normalize(dtm)

# Cluster with K-Means
kmeans = KMeans(n_clusters=n_clusters).fit(X)

# Clustring Results
clusters = kmeans.labels_
centers = kmeans.cluster_centers_

# Insert Result Column
df['cluster'] = clusters

for i in range(n_clusters):

    #Select row with given cluster
    cdf = df.loc[df['cluster'] == i]  

    #Calculate frequent words : sum of rows to vector
    count_vector = dtm[cdf.index].sum(axis=0).getA().ravel()

    #Calculate Distance
    center = sparse.csr_matrix(centers[i])
    dist_vectors = pairwise_distances(X[cdf.index], center).ravel()

    cluster_keywords = []
    if count_vector is not None:
        word_indexes = count_vector.argsort()[::-1][:n_cluster_freqwords] #top words
        for word_index in word_indexes:
            keyword = words[word_index]
            count = count_vector[word_index]
            cluster_keywords.append('{}({})'.format(keyword, count))

    cluster_keywords = ','.join(cluster_keywords)

    print("cluster:{} articles:{} keywords:{}".format(i+1, len(cdf), cluster_keywords))

    cluster_titles = cdf['title'].values.tolist()
    articles = ['[{:.2f}]{}'.format(dist_vectors[i],cluster_titles[i]) for i in range(len(cluster_titles))]
    title_list = '\n '.join(articles)

    print(' {}'.format(title_list))
    print()

print('title:{}, cluster:{}'.format(len(titles), n_clusters))
