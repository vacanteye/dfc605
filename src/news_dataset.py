import sqlite3, pdb
import pandas as pd
from os import path

def fetch_news_samples(date, count):
    if len(date) != 8:
        return None

    fname = '../dat/news_daily_{}.db'.format(date)
    if not path.exists(fname):
        return None

    conn = sqlite3.connect(fname)
    sql = 'SELECT date,title,nouns FROM news_table LIMIT {}'.format(count)
    df = pd.read_sql_query(sql, conn)
    conn.close()

    return df
