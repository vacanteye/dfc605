import sqlite3
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
    
# Load Data
#stopwords = read_dic('../dat/stopwords_ko.txt')

conn = sqlite3.connect('../dat/news_daily_20200501.db')
#df = pd.read_sql_query('SELECT title,nouns FROM news_table LIMIT 100', conn)
df = pd.read_sql_query('SELECT title,nouns FROM news_table', conn)
conn.close()

titles = df['title'].tolist()
docs = df['nouns'].tolist()

# Vectorizing
vectorizer = CountVectorizer()

# Clustering
X = vectorizer.fit_transform(docs)

# Features
names = vectorizer.get_feature_names()

# L2 normalizing
X = normalize(X)

n_clusters = 50

# Cluster with k-means
y_kmeans = KMeans(n_clusters=n_clusters).fit(X)

# trained labels and cluster centers

clusters = y_kmeans.labels_.tolist()
centroids = y_kmeans.cluster_centers_.tolist()

articles = {'title': titles, 'cluster': clusters}
df = pd.DataFrame(articles, index=[clusters], columns=['title', 'cluster'])


for i in range(n_clusters):
    sub_frame = df.loc[df['cluster'] == i]
    print("Cluster:{} Size:{}".format(i+1, len(sub_frame)))
    title_list = '\n '.join(sub_frame['title'].values.tolist())
    print(' {}'.format(title_list))
    print()

print()
print('shape=', X.shape)
print('title:{}, cluster:{}'.format(len(titles), n_clusters))

'''
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
model = model.fit(X.todense())
plot_dendrogram(model, truncate_mode='level', p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()
'''


