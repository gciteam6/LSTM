
# coding: utf-8

# ## **発電量データを30分単位に変更する**

# In[7]:


# データ加工・処理・分析モジュール
import numpy as np
import pandas as pd


# In[8]:


# train_kwhをエクセル等で開くとdatetimeが指数表示に直される可能性がある
# その場合うまくいかないので201201010120の形になってることを確認する必要あり
output_data = pd.read_csv('train_kwh.tsv', delimiter = '\t')


# In[9]:


def set_time(dataframe, col_name):
    '''
    to_datetimeを使うための前処理
    '''
    dataframe[col_name] = dataframe[col_name].map(lambda x : transform_time(x))
    return dataframe


# In[10]:


def transform_time(x):
    '''
    set_time内で使う関数
    to_datetimeで24時をサポートしないので00に変更する処理
    '''
    str_x = str(x)
    res = ''
    if str(x)[8:10] == '24':
        res = str_x[0:4] + '-' + str_x[4:6] + '-' + str_x[6:8] + ' 00:'+str_x[10:12] 
    else:
        res = str_x[0:4] + '-' + str_x[4:6] + '-' + str_x[6:8] + ' '+ str_x[8:10] +':'+str_x[10:12]
    return res


# In[11]:


# datetimeの行をpd.Timestampのインスタンスに変更
output_data = set_time(output_data, 'datetime')
output_data['datetime'] = output_data['datetime'].map(lambda x : pd.to_datetime(x))

# 30分ごとに集計
output_data.set_index('datetime').groupby(pd.TimeGrouper(freq='1800s', closed='left')).sum()

