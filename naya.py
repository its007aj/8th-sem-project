import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from ast import literal_eval
data = pd.read_csv("goibibo.csv")
print(data.head())
print(data.state.unique())
print(data.City.unique())

def impute(column):
    column = column[0]
    if (type(column) != list):
        return "".join(literal_eval(column))
    else:
        return column
#print(type(data['hotel_facilities']))
#for i in range(5000):
#    data['hotel_facilities'][i]=data['hotel_facilities'][i].tolist()
#data["hotel_facilities"] = data[["hotel_facilities"]].apply(impute, axis=1)
#print(data['hotel_facilities'])
data['tags']=data['additional_info']+data['hotel_facilities']
data['tags']=data['tags'].apply(str)
#data["tags"] = data[["tags"]].apply(impute, axis=1)
print(data['tags'])
data['City'] = data['City'].str.lower()
data['tags'] = data['tags'].str.lower()
print(data['tags'])
data['Attractions'] = data['Attractions'].fillna("local area")

def recommend_hotel(location, description):
    description = description.lower()
    word_tokenize(description)
    stop_words = stopwords.words('english')
    lemm = WordNetLemmatizer()
    filtered  = {word for word in description if not word in stop_words}
    filtered_set = set()
    for fs in filtered:
        filtered_set.add(lemm.lemmatize(fs))
    country = data[data['City']==location.lower()]
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
    return country[["Hotel_Name", "Ratings", "Hotel_Address","Attractions"]].head()
#print(recommend_hotel('Bhilwara', 'beverage restaurant'))
def predict(where,why):
   return recommend_hotel(where,why)