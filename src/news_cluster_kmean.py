import sqlite3, pdb
import pandas as pd
import numpy as np
import news_dataset

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from scipy import sparse

from soyclustering import SphericalKMeans

# Static Configuration
n_samples = 100
n_clusters = 5
verbose = 0

max_iter = 300
n_top_words = 7
ngram_range = (1,2)

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
        print('')
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
        print('')
        exit(1)

   
# Load Data
df = news_dataset.fetch_news_samples('20200501', n_samples)

titles = df['title'].tolist()
docs = df['nouns'].tolist()
#docs = df['title'].tolist()

# Document-Term Matrix
dtm = vectorizer.fit_transform(docs)

# Word List
words = vectorizer.get_feature_names()

# L2 Normalizing
X = normalize(dtm) if i_metric == 1 else dtm

# Document Clustring
model.fit(X)

# Clustring Results
clusters = model.labels_
centers = model.cluster_centers_

# Insert Result Column
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
