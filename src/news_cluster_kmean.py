'''
    Brief : News Summarization using K-Means Clustering
    Author: Dong-Jun Kim, DFC605 2020
'''

import sqlite3, pdb, sys
import pandas as pd
import numpy as np
import news_dataset

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy import sparse

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import TruncatedSVD

from sklearn.cluster import KMeans          #Euclidian Distance (Least Squared)
from soyclustering import SphericalKMeans   #Cosine Distance

# Static Configuration
n_samples = 2000
n_clusters = 150
verbose = 0

max_iter = 300
n_top_words = 7
ngram_range = (1,1)

# Conditional Configurations
i_metric = 1
i_vectorizer = 1
model = None
vectorizer = None

# Select Distance Metric
while True:
    try:
        i_metric = int(input("Select vector distance metic {1:Euclidian, 2:Cosine}: "))
        if i_metric == 1:
            model = KMeans(
                n_clusters=n_clusters,
                max_iter = max_iter,
                verbose = verbose
            )
            break
        elif i_metric == 2:
            model = SphericalKMeans(
                n_clusters=n_clusters,
                max_iter=max_iter,
                init='similar_cut',
                sparsity='minimum_df',
                minimum_df_factor=0.05,
                verbose=verbose
            )
            break
        else:
            print('Invalid value')
            continue
    except:
        print(sys.exc_info())
        exit(1)


# Select Document Vectorizer
while True:
    try:
        i_vectorizer = int(input("Select Vectorizer {1:CountVectorizer, 2:TF-IDF}: "))
        if i_vectorizer == 1:
            vectorizer = CountVectorizer(
                ngram_range=ngram_range
            )
            break
        elif i_vectorizer == 2:
            vectorizer = TfidfVectorizer(
                ngram_range=ngram_range
            )
            break
        else:
            print('Invalid value')
            continue
    except:
        print(sys.exc_info())
        exit(2)

   
# Load Data
df = news_dataset.fetch_news_samples('20200501', n_samples)

titles = df['title'].tolist()
docs = df['nouns'].tolist()
#docs = df['title'].tolist()

# Document-Term Matrix for Word Counting
dtm = CountVectorizer().fit_transform(docs)

# Selected Vectorizer
X = vectorizer.fit_transform(docs)

# Generate Visualization Info using Dimension Redunction
svd = TruncatedSVD(n_components=3)
pos = svd.fit_transform(X)
x_pos = pos[:,0]
y_pos = pos[:,1]
z_pos = pos[:,2]

# Word List
words = vectorizer.get_feature_names()

# L2 Normalizing
if i_vectorizer == 1:
    X = normalize(X)

# Document Clustring
model = model.fit(X)

# Clustring Results
clusters = model.labels_
centers = model.cluster_centers_

# Result Columns
df['cluster'] = clusters
df['distance'] = 0

metric = 'euclidean' if i_metric == 1 else 'cosine'

for i in range(n_clusters):

    #Select row with given cluster
    cdf = df.loc[df['cluster'] == i].sort_index(ascending=False)  

    #Calculate frequent words : sum of rows to vector
    count_vector = dtm[cdf.index].sum(axis=0).getA().ravel()

    #Calculate Distance
    center = sparse.csr_matrix(centers[i])
    dist_vectors = pairwise_distances(X[cdf.index], center, metric=metric).ravel()
    cdf['distance'] = dist_vectors

    cluster_topwords = []
    if count_vector is not None:
        word_indexes = count_vector.argsort()[::-1][:n_top_words] #top words
        for word_index in word_indexes:
            keyword = words[word_index]
            count = count_vector[word_index]
            cluster_topwords.append('{}({})'.format(keyword, count))

    cluster_topwords = ','.join(cluster_topwords)

    # Cluster Infos
    print("Cluster:{} Articles:{} Topwords:{}".format(i+1, len(cdf), cluster_topwords))

    cluster_titles = ['{:.2f} - {} - {}'.format(row['distance'], row['date'], row['title']) for i, row in cdf.iterrows()]
    cluster_titles = '\n '.join(cluster_titles)
    print(' {}'.format(cluster_titles))
    print()

print('title:{}, cluster:{}'.format(len(titles), n_clusters))

'''
# 2D Graph
plt.scatter(x_pos, y_pos, c=clusters, cmap='viridis')
plt.show()
'''

# 3D Graphs
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.scatter(x_pos, y_pos, z_pos, c = clusters, cmap='viridis')
ax.scatter(x_pos, y_pos, z_pos, c = clusters, cmap='hsv')
plt.show()

