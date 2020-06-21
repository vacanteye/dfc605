import sqlite3
import pandas as pd
from konlpy.tag import Kkma
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

kkma = Kkma()

def preprocessing(doc):
    nouns = kkma.nouns(doc)
    return ' '.join(nouns)
    
#Load DB 
conn = sqlite3.connect('../dat/news_daily_20200501.db')
df = pd.read_sql_query('SELECT * FROM news_table', conn)
conn.close()

df['title'] = df['title'].apply(preprocessing)
docs = df['title'].tolist()

# vectorizing
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)

print('feagure names:{}'.format(len(vectorizer.get_feature_names())))

# L2 normalizing
X = normalize(X)

# training k-means
kmeans = KMeans(n_clusters=100).fit(X)

# trained labels and cluster centers
labels = kmeans.labels_
centers = kmeans.cluster_centers_

print('labels', labels)
print('centers', centers)


