import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data=pd.read_csv('goibibo.csv')
data['tags']=data['additional_info']+data['hotel_facilities']
data['Attractions'] = data['Attractions'].fillna("local area")

def cosine(city, hotelname):
    city = city.title()
    hotelname = hotelname.title()
    #print(hotelname)
    tf_idf = TfidfVectorizer()
    df_hotel_tf_idf_described = tf_idf.fit_transform(data['tags'].values.astype('U'))
    m2m = cosine_similarity(df_hotel_tf_idf_described)
    df_tfidf_m2m = pd.DataFrame(cosine_similarity(df_hotel_tf_idf_described))
    index_to_hotel_id = data['Hotel_Name']
    df_tfidf_m2m.columns = [str(index_to_hotel_id[int(col)]) for col in df_tfidf_m2m.columns]
    df_tfidf_m2m.index = [index_to_hotel_id[idx] for idx in df_tfidf_m2m.index]
    df_tfidf_m2m.iloc[0].sort_values(ascending=False)[:10]
    first_column = df_tfidf_m2m.iloc[:, 0]
    new_df=df_tfidf_m2m[hotelname].sort_values(ascending=False)
    data['City'] = data['City'].str.lower()
    country = data[data['City']==city.lower()]
    country = country.set_index(np.arange(country.shape[0]))
    #print(country)
    #for hotel,cosine in new_df
    #print(new_df.index)
    df=pd.DataFrame(new_df.index)
    #print("*****************************************")
    #print(df)
    #country.where[country.values==df.values]
    df.columns =['Hotel_Name']
    merge=pd.merge(country,df, on=['Hotel_Name'],how='inner')
    #print("*****************************************last********************************")
    merge.sort_values('Ratings', ascending=False, inplace=True)
    merge.reset_index(inplace=True)
    #merge.index=np.arange(1,len(country)+1)
    #print(df)
    return merge[["Hotel_Name", "Ratings", "Hotel_Address","Attractions"]].head().drop_duplicates()

def intersection(city, description):
    data['tags']=data['tags'].apply(str)
    #print(data['tags'])
    data['City'] = data['City'].str.lower()
    data['tags'] = data['tags'].str.lower()
    #print(data['tags'])
    description = description.lower()
    word_tokenize(description)
    stop_words = stopwords.words('english')
    lemm = WordNetLemmatizer()
    filtered  = {word for word in description if not word in stop_words}
    filtered_set = set()
    for fs in filtered:
        filtered_set.add(lemm.lemmatize(fs))
    country = data[data['City']==city.lower()]
    country = country.set_index(np.arange(country.shape[0]))
    cos = []
    for i in range(country.shape[0]):
        temp_token = word_tokenize(country["tags"][i])
        temp_set = [word for word in temp_token if not word in stop_words]
        temp2_set = set()
        for s in temp_set:
            temp2_set.add(lemm.lemmatize(s))
        vector = temp2_set.intersection(filtered_set)
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

def predict(city, hotelname, description):
    df1 = df2 = pd.DataFrame()
    if hotelname!='':
        df1 = cosine(city, hotelname)
    if description!='':
        df2 = intersection(city, description)
    d = pd.concat([df1, df2]).head().drop_duplicates()
    d.index= np.arange(1, len(d)+1)
    return d




