import sqlite3, pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

# Configurations
n_init = 1
max_iter = 300
n_data_size = 2000
n_cluster_freqwords = 7
ngram_range = (1,2)

# Load Data
conn = sqlite3.connect('../dat/news_daily_20200501.db')
df = pd.read_sql_query('SELECT date,title,nouns FROM news_table LIMIT {}'.format(n_data_size), conn)
conn.close()

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
ks = range(20, 200+1, 20)
for n_clusters in ks:
    clusterer = KMeans(
        n_init = n_init,
        n_clusters=n_clusters, 
        verbose=1,
        max_iter=max_iter
    )

    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)

    print("For n_clusters =", n_clusters,
        "The average silhouette_score is :", silhouette_avg) 
    print()

    inertias.append(clusterer.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()
#plt.savefig('../../assets/images/markdown_img/kmean_clustering_20180513.svg')

