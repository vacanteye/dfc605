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
from sklearn.decomposition import PCA, NMF, LatentDirichletAllocation 

from sklearn.cluster import KMeans          #Euclidian Distance
from soyclustering import SphericalKMeans   #Cosine Distance

def get_topics(model, feature_names, n_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        #message = "Topic #{}: ".format(topic_idx+1)
        message = ''
        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        topics.append(message)
    return topics

def log(fp, message):
    print(message)
    fp.write(message)
    fp.write('\n')

# Static Configuration
n_samples = 200
n_clusters = 20
verbose = 0

max_iter = 300
n_top_words = 5
n_topics = 1
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

# Word List
words = vectorizer.get_feature_names()

# L2 Normalizing
if i_vectorizer == 1:
    X = normalize(X)

# Generate Visualization Info using Dimension Redunction
'''
svd = TruncatedSVD(n_components=3)
pos = svd.fit_transform(X)
'''
pca = PCA(n_components=3)
pos = pca.fit_transform(X.todense())
x_pos = pos[:,0]
y_pos = pos[:,1]
z_pos = pos[:,2]

# Document Clustring
model = model.fit(X)

# Clustring Results
clusters = model.labels_
centers = model.cluster_centers_

# Result Columns
df['cluster'] = clusters
df['distance'] = 0
df['topic'] = 0

metric = 'euclidean' if i_metric == 1 else 'cosine'

figname = '../out/fig_{}_{}.png'.format(i_metric, i_vectorizer)
logname = '../out/out_{}_{}.log'.format(i_metric, i_vectorizer)
csvname = '../out/out_{}_{}.csv'.format(i_metric, i_vectorizer)

flog = open(logname, 'w')
fcsv = open(csvname, 'w')

fcsv.write('Cluster,Topic,Title\n')

for i in range(n_clusters):

    #Select row with given cluster
    cdf = df.loc[df['cluster'] == i].sort_index(ascending=False)  

    #Top words using Word Countring
    count_vector = dtm[cdf.index].sum(axis=0).getA().ravel()

    #Top words using LDA
    cluster_samples = X[cdf.index]
    lda = LatentDirichletAllocation(
        n_components=n_topics, max_iter=max_iter,
        learning_method='online',
        learning_offset=50.,
        random_state=0,
        verbose=0
    )
    lda.fit(cluster_samples)
    topics = get_topics(lda, words, n_top_words)
    topics = '\n '.join(topics)
    cdf['topic'] = topics.splitlines()[0]

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
            #cluster_topwords.append('{}({})'.format(keyword, count))
            cluster_topwords.append('{}'.format(keyword, count))

    cluster_topwords = ' '.join(cluster_topwords)

    # Cluster Infos
    log(flog, 'Cluster #{}: {} article(s)'.format(i+1, len(cdf)))
    log(flog, '  Words Counting:{}'.format(cluster_topwords))
    log(flog, '  LDA Topics    :{}'.format(topics))

    cluster_titles = ['{:.2f}-{}'.format(row['distance'], row['title']) 
        for i, row in cdf.iterrows()]

    #cluster_titles = '\n    '.join(cluster_titles[:5])
    cluster_titles = '\n    '.join(cluster_titles)
    log(flog, '    {}'.format(cluster_titles))
    log(flog, '    ...')
    log(flog,'')

    csv_lines = ['{},{},{}\n'.format(row['cluster'], row['topic'], row['title'].replace(',',' ')) 
        for i, row in cdf.iterrows()]

    for line in csv_lines:
        fcsv.write(line)

log(flog, 'title:{}, cluster:{}'.format(len(titles), n_clusters))

flog.close()
fcsv.close()

# 3D Graphs
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_pos, y_pos, z_pos, c = clusters, cmap='viridis')
plt.savefig(figname)

