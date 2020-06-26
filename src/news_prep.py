'''
    Brief : News Dataset Preprocessing
    Author: Dong-Jun Kim, DFC605 2020
'''
import sqlite3, glob, os, re
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from konlpy.tag import Kkma     #Very Slow
from konlpy.tag import Okt      #Normal
from konlpy.tag import Mecab    #Normal
from konlpy.tag import Hannanum

tokenizer = Hannanum()
stopwords = []

# 
def remove_stopwords(text):
    if len(stopwords) == 0:
        fp = open('../dat/stopwords_ko.txt', 'r')
        for line in fp:
            stopwords.append(line)
    for word in stopwords:
        text = text.replace(word, '')

#
def strip_tag(text, stag, etag):
    #Remove Related Articles
    while True:
        spos = text.find(stag)
        epos = text.find(etag, spos+1)
        if spos == -1 or epos == -1:
            break
        text = text[:spos] + text[epos+len(etag):]
    return text

#
def get_refined_content(title, content):
    #Remove Prefix    
    pos = content.find('text://')
    if pos != -1:
        content = content[7:]

    #Remove Relevant Articles
    content = strip_tag(content, '<!-- r_start', 'r_end //-->')

    #Remove 'a' Tags
    soup = BeautifulSoup(content, 'html.parser')
    for s in soup.select('a'):
        s.extract()

    content = soup.get_text()

    # Remove Source & Reporter
    content = strip_tag(content, '[', ']')

    #Remove Special Characters
    #content = re.sub('[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9]', ' ', content).strip()
    content = re.sub('[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z]', ' ', content).strip()

    #Use title if content marked empty
    if content.startswith('내용이 없습니다') or len(content) == 0:
        content = title
        content = strip_tag(content, '[', ']')

    return content

#
def get_daily_periods(dates):
    periods = []
    for date in dates:
        date_str = date.strftime('%Y%m%d')
        periods.append((date_str, date_str))
    return periods

#
def get_weekly_periods(dates):
    # Day of Week  : datetime.weekday()
    # Week of Year : datetime.isocalendar()[1]
    periods = []
    sdate = None
    pdate = None
    for date in dates:
        if sdate == None:
            sdate = date
            pdate = date
            continue

        sweek = sdate.isocalendar()[1] 
        week = date.isocalendar()[1]

        if sweek != week:
            periods.append((sdate.strftime('%Y%m%d'), pdate.strftime('%Y%m%d')))
            sdate = date

        pdate = date

    # Remainder
    if sdate != pdate:
        periods.append((sdate.strftime('%Y%m%d'), pdate.strftime('%Y%m%d')))

    return periods


#
#Input Period
while True:
    try:
        inp_period = int(input("Select Period [1:Daily 2:Weekly]: "))
        if inp_period in (1, 2):
            break
        else:
            print('Invalid Period')
            continue
    except:
        print('')
        exit(1)

# DB Connection
#try:
DATABASE = '../dat/news.db'
conn = sqlite3.connect(DATABASE)
sql = 'SELECT DISTINCT date FROM news_table ORDER BY date'
df = pd.read_sql_query(sql, conn)
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df.set_index('date', inplace=True)

dates = df.index.tolist()
periods = []
if inp_period == 1:
    periods = get_daily_periods(dates)
elif inp_period == 2:
    periods = get_weekly_periods(dates)

for period in periods:
    sql = '''SELECT dttm, titl, sour, cod2, data 
            FROM news_table 
            WHERE date >= {} AND date <= {} ORDER BY dttm'''.format(period[0], period[1])

    df = pd.read_sql_query(sql, conn)
    df.rename(columns = {'dttm': 'date', 
                         'titl': 'title', 
                         'cod2': 'code', 
                         'sour': 'source',   #Source Name
                         'data': 'content'}, inplace=True)

    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d%H%M%S')
    df.insert(5, 'plain', '')
    df.insert(6, 'nouns', '')

    for index, row in df.iterrows():
        plain_content = get_refined_content(row['title'], row['content'])
        df.loc[index, 'plain'] = plain_content
        df.loc[index, 'nouns'] = ' '.join(tokenizer.nouns(plain_content))
        #print(index, row['title'])

    fname = ''
    if period[0] == period[1]:
        fname = '../dat/news_daily_{}.db'.format(period[0])
    else:
        fname = '../dat/news_weekly_{}-{}.db'.format(period[0], period[1])

    #Remove file if exists
    if os.path.isfile(fname):
        os.remove(fname)

    conn_out = sqlite3.connect(fname)
    df.to_sql('news_table', conn_out)
    conn_out.close()

    print('DONE: {}'.format(fname)) 

if conn: 
    conn.close()

