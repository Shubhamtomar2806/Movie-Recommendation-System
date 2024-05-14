#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


movies=pd.read_csv("tmdb_5000_movies.csv")
credits=pd.read_csv("tmdb_5000_credits.csv")


# In[3]:


movies.head(1)


# In[4]:


credits.head(1)["cast"]


# In[5]:


movies=movies.merge(credits,on="title")   # merge dataframe on tthe basis of title column


# In[6]:


movies.head(1)


# In[7]:


# genres , Id, keywwords, title, overview, cast, crew #  seprate these columns from the dataframe and assign bback tto the movies
movies=movies[["movie_id", "title", "genres", "keywords", "overview", "cast", "crew"]]   


# In[8]:


movies.head(1)

# preprocicng
# In[9]:


movies.isnull().sum()    # check there is any null value present in this data frame or not.m


# In[10]:


movies.dropna(inplace=True)    # remove null values from the data frame 


# In[11]:


movies.duplicated().sum()     # if duplicate rows found then we use this coodde to remove duplicate rows 
                              # movies.drop_duuplicates(inplace-True)


# In[12]:


movies.iloc[0].genres


# In[13]:


#'[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
# ['Action','advanture','Fantasy','science fiction']


# In[14]:


import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
)


# In[15]:


def convert(obj):
    l=[]
    for i in ast.literal_eval(obj):    # ast.literal_eval is used to convert tthe string into a list too perform loop
        l.append(i['name'])
    return l


# In[16]:


movies['genres']=movies['genres'].apply(convert)


# In[17]:


movies.head(1)


# In[18]:


movies.iloc[0].keywords


# In[19]:


movies['keywords']=movies['keywords'].apply(convert)


# In[20]:


movies.head(1)


# In[21]:


def convert3(obj):
    l=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
            l.append(i['name'])
            counter+=1
        else:
            break
    return l
        
        
        
        


# In[22]:


movies['cast']=movies['cast'].apply(convert3)


# In[23]:


movies.head(11)


# In[24]:


movies['crew'][0]


# In[25]:


def fetch_director(obj):
    l=[]
    for i in ast.literal_eval(obj):
        if i['job']== 'Director':
            l.append(i['name'])
            break
    return l
        


# In[26]:


movies['crew']=movies['crew'].apply(fetch_director)


# In[27]:


movies.head(1)


# In[28]:


movies['overview'][0]


# In[29]:


movies['overview']=movies['overview'].apply(lambda x:x.split()) # convert  thhe string availabble inn the oveerview column into a list 


# In[30]:


movies.head(1)


# In[31]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","")  for i in x])


# In[32]:


movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","")  for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","")  for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","")  for i in x])


# In[33]:


movies.head()


# In[34]:


movies['tags']=movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# mmovies.head(

# In[35]:


movies.head()


# In[36]:


new_df = movies[['movie_id','title','tags']]


# In[37]:


new_df.head()


# In[38]:


new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))  # convert list in tags column into a string


# In[39]:


new_df.head()


# In[40]:


import nltk


# In[41]:


from nltk.stem.porter import  PorterStemmer
ps= PorterStemmer()


# In[42]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
     
    return " ".join(y)
    
        


# In[43]:


new_df["tags"]=new_df["tags"].apply(stem)


# In[44]:


new_df['tags'][0]


# In[45]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[46]:


new_df.head(1)


# In[47]:


from sklearn.feature_extraction.text  import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')


# In[48]:


vectors=cv.fit_transform(new_df['tags']).toarray()


# In[49]:


vectors[0]


# In[50]:


cv.get_feature_names_out()


# In[51]:


from sklearn.metrics.pairwise import cosine_similarity


# In[52]:


similarity=cosine_similarity(vectors)


# In[53]:


sorted(list(enumerate(similarity[0])),reverse =True, key=lambda x:x[1])[1:60]


# In[54]:


def recommend(movie):
    movie_index = new_df[new_df['title']== movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse = True, key = lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[55]:


recommend('Avatar')


# In[56]:


import pickle


# In[57]:


pickle.dump(new_df, open('movies.pkl','wb'))


# In[60]:


new_df['title'].values


# In[62]:


pickle .dump(new_df.to_dict,open('movie_dict.pkl','wb'))


# In[ ]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




