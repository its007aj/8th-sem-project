#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


data=pd.read_csv('goibibo.csv')


# In[3]:


data['tags']=data['additional_info']+data['hotel_facilities']
data['Attractions'] = data['Attractions'].fillna("local area")


# In[4]:


def cosine(city, hotelname):
    city = city.title()
    hotelname = hotelname.title()
    tf_idf = TfidfVectorizer()
    df_hotel_tf_idf_described = tf_idf.fit_transform(data['tags'].values.astype('U'))
    df_tfidf_m2m = pd.DataFrame(cosine_similarity(df_hotel_tf_idf_described))
    index_to_hotel_id = data['Hotel_Name']
    df_tfidf_m2m.columns = [str(index_to_hotel_id[int(col)]) for col in df_tfidf_m2m.columns]
    df_tfidf_m2m.index = [index_to_hotel_id[idx] for idx in df_tfidf_m2m.index]
    print(df_tfidf_m2m.head())
    new_df=df_tfidf_m2m[hotelname].sort_values(ascending=False)
    data['City'] = data['City'].str.lower()
    country = data[data['City']==city.lower()]
    country = country.set_index(np.arange(country.shape[0]))#dataset hotel of particular city
    df=pd.DataFrame(new_df.index)
    
    
    df.columns =['Hotel_Name']#cosine similarity hotels
    merge=pd.merge(country,df, on=['Hotel_Name'],how='inner')
    #print("*****************************************last********************************")
    merge.sort_values('Ratings', ascending=False, inplace=True)
    merge.reset_index(inplace=True)
    #merge.index=np.arange(1,len(country)+1)
    
    #print(df)
    return merge[["Hotel_Name", "Ratings", "Hotel_Address","Attractions"]].head().drop_duplicates()


# In[5]:


def intersection(city, description):
    data['tags']=data['tags'].apply(str)
    #print(data['tags'])
    data['City'] = data['City'].str.lower()
    data['tags'] = data['tags'].str.lower()
    #print(data['tags'])
    data['Attractions'] = data['Attractions'].fillna("local area")
    description = description.lower()
    word_tokenize(description)
    stop_words = stopwords.words('english')
    lemm = WordNetLemmatizer()
    filtered  = {word for word in description if not word in stop_words}
    filtered_set = set()
    for fs in filtered:
        filtered_set.add(lemm.lemmatize(fs))#description
    country = data[data['City']==city.lower()]
    country = country.set_index(np.arange(country.shape[0]))
    cos = []
    for i in range(country.shape[0]):
        temp_token = word_tokenize(country["tags"][i])
        temp_set = [word for word in temp_token if not word in stop_words]
        temp2_set = set()
        for s in temp_set:
            temp2_set.add(lemm.lemmatize(s))#tags
        vector = temp2_set.intersection(filtered_set)#intersection of tags and description
        cos.append(len(vector))
    
    country['similarity']=cos
    country = country.sort_values(by='similarity', ascending=False)
    country.drop_duplicates(subset='Hotel_Name', keep='first', inplace=True)
    country.sort_values('Ratings', ascending=False, inplace=True)
    country.reset_index(inplace=True)
    country.index=np.arange(1,len(country)+1)
    #print(country.to_string(index = False))
    #print(country.style.hide_index()
    return country[["Hotel_Name", "Ratings", "Hotel_Address","Attractions"]].head().drop_duplicates()


# In[9]:


def predict(city, hotelname, description):
    df1 = df2 = pd.DataFrame()
    if hotelname!='':
        df1 = cosine(city, hotelname)
    if description!='':
        df2 = intersection(city, description)
    result=pd.concat([df1, df2]).head().drop_duplicates()
    result.sort_values('Ratings', ascending=False, inplace=True)
    result.reset_index(inplace=True)
    result.index= np.arange(1, len(result)+1)
    return result[["Hotel_Name", "Ratings", "Hotel_Address","Attractions"]].head()


# In[10]:


d = predict('bangalore', 'the tavern', 'business')

print(d)


# In[ ]:





# In[ ]:





# In[ ]:




