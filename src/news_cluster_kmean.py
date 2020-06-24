import sqlite3, pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from konlpy.tag import Hannanum
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

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
n_clusters = 50
n_data_size = 2000
    
# Load Data
conn = sqlite3.connect('../dat/news_daily_20200501.db')
df = pd.read_sql_query('SELECT title,nouns FROM news_table LIMIT {}'.format(n_data_size), conn)
conn.close()

titles = df['title'].tolist()
#docs = df['nouns'].tolist()
docs = df['title'].tolist()

# Vectorizing
vectorizer = CountVectorizer()

# Clustering
bows = vectorizer.fit_transform(docs) # Bag of Words

# Features
words = vectorizer.get_feature_names()

# L2 normalizing
X = normalize(bows)

# Cluster with K-Means
kmeans = KMeans(n_clusters=n_clusters).fit(X)

# trained labels and cluster centers
clusters = kmeans.labels_
centers = kmeans.cluster_centers_


articles = {'title': titles, 'cluster': clusters.tolist()}
df = pd.DataFrame(articles, index=[clusters], columns=['title', 'cluster'])

for i in range(n_clusters):

    #select row with given cluster
    cluster_frame = df.loc[df['cluster'] == i]  

    #Frequent Words
    cluster_indexes = np.where(clusters == i)
    if len(cluster_indexes) != 0: cluster_indexes = cluster_indexes[0] #tuple to ndarray

    count_vector = None
    for index in cluster_indexes:
        vector = bows[index].todense().getA().ravel()
        count_vector = vector + vector if count_vector is not None else vector

    cluster_keywords = []
    if count_vector is not None:
        word_indexes = count_vector.argsort()[::-1][:7] #top 5 word indexes
        for word_index in word_indexes:
            keyword = words[word_index]
            count = count_vector[word_index]
            cluster_keywords.append('{}({})'.format(keyword, count))

    cluster_keywords = ','.join(cluster_keywords)

    print("cluster:{} articles:{} keywords:{}".format(i+1, len(cluster_frame), cluster_keywords))

    title_list = '\n '.join(cluster_frame['title'].values.tolist())
    print(' {}'.format(title_list))
    print()

print('title:{}, cluster:{}'.format(len(titles), n_clusters))

