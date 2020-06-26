'''
    Brief : Clustering Count Evaluation
    Author: Dong-Jun Kim, DFC605 2020
'''
import sqlite3, pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import news_dataset

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from soyclustering import SphericalKMeans   #Cosine Distance


# Configurations
n_init = 1
max_iter = 50
n_samples = 2000
n_cluster_freqwords = 7
ngram_range = (1,1)

# Load Data
df = news_dataset.fetch_news_samples('20200501', n_samples)
docs = df['nouns'].tolist()

# Vectorizing
vectorizer = CountVectorizer(
    ngram_range=ngram_range
)

# Document-Term Matrix
dtm = vectorizer.fit_transform(docs)

# L2 Normalizing
X = normalize(dtm)

inertias = []
scores = []
ks = range(50, 1001, 50)
for n_clusters in ks:

    clusterer = SphericalKMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
        init='similar_cut',
        sparsity='minimum_df',
        minimum_df_factor=0.05,
        verbose=1
    )

    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)

    print("For n_clusters =", n_clusters,
        "The average silhouette_score is :", silhouette_avg) 
    print()

    inertias.append(clusterer.inertia_)
    scores.append(silhouette_avg)
    
# Plot ks vs inertias
fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_xlabel('# of Clusters')
ax1.set_ylabel('Cluster Inertia', color=color)
ax1.plot(ks, inertias, '-o', color=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('The Average Silhouette Score', color=color)
ax2.plot(ks, scores, '-s', color=color)

fig.tight_layout()
plt.show()
#plt.savefig('../../assets/images/markdown_img/kmean_clustering_20180513.svg')

