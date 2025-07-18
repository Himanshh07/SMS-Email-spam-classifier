#!/usr/bin/env python
# coding: utf-8

# In[76]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('spam.csv', encoding='latin1')
df.sample(5)


# In[3]:


df.shape


# In[4]:


df.isnull().sum()


# In[5]:


df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace=True)


# In[6]:


df.sample(5)


# In[7]:


df.isnull().sum()


# In[8]:


#Rename the columns
df.rename(columns={'v1':'target', 'v2':'text'}, inplace=True)


# In[9]:


# df.sample()
#Checking for the duplicate value in dataset
df.duplicated().sum()


# In[10]:


#Remove the duplicate value from the dataset
df = df.drop_duplicates(keep='first')


# In[11]:


df.duplicated().sum()


# In[12]:


# changing the value of ham and spam with integer value
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[13]:


df['target'] = encoder.fit_transform(df['target'])


# In[14]:


# df.sample(5)
df.shape


# ## EDA

# In[15]:


df['target'].value_counts()


# In[16]:


#Show the representation in percentage using graph 
import matplotlib.pyplot as plt


# In[17]:


plt.pie(df['target'].value_counts(), labels=['ham', 'spam'], autopct='%0.2f')
plt.show()


# In[18]:


#install the NLTK library
get_ipython().system('pip install nltk')


# In[19]:


import nltk 


# In[20]:


nltk.download('punkt')
nltk.download('punkt_tab')


# In[21]:


#Creating a new column that stores the value of number of words in text
df['num_character'] = df['text'].apply(len)


# In[22]:


#Creating a new column that stores a number of words in text
df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
df.head()


# In[23]:


# creating a new column to store the value of number of sentences in text
df['num_sentence'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))


# In[24]:


df[['num_character','num_words','num_sentence']].describe()
# df.columns


# In[25]:


# Describe of ham sms in data
df[df['target']==0][['num_character','num_words','num_sentence']].describe()


# In[26]:


# Describe of spam sms in data
df[df['target']==1][['num_character','num_words','num_sentence']].describe()


# In[27]:


#Plotting the num_character in ham sms with help if histogram 
sns.histplot(df[df['target']==0]['num_character'])
#Plotting the num_character in spam sms with help if histogram 
sns.histplot(df[df['target']==1]['num_character'], color='red')


# In[28]:


sns.histplot(df[df['target']==0]['num_words']) 
sns.histplot(df[df['target']==1]['num_words'], color='red')
# plt.figure(figsize=(11,6))


# ## Data Preprocessing

# In[29]:


from nltk.corpus import stopwords
import string


# In[35]:


# Connvert the string into lowercase
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    #for removing special chracter 
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)


    # for removing the punctuation and stopwords
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    #for stemming
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)




# In[36]:


transform_text("Hellooooo Howww Are %% you Himasnhu hello dancing hello")


# In[34]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ps.stem('Dancing')


# In[41]:


# Adding the transformed data into new column 
df['transformed'] = df['text'].apply(transform_text)
df.head()


# In[ ]:


get_ipython().system('pip install --user wordcloud')


# In[32]:


import sys
get_ipython().system('{sys.executable} -m pip install wordcloud --user')


# In[33]:


from wordcloud import WordCloud
print("WordCloud imported successfully!")


# In[48]:


wc = WordCloud(width=500, height=500, min_font_size=10, background_color='White')


# In[49]:


spam_wc = wc.generate(df[df['target']==1]['transformed'].str.cat(sep=''))
plt.imshow(spam_wc)


# In[52]:


ham_wc = wc.generate(df[df['target']==0]['transformed'].str.cat(sep=''))
plt.imshow(ham_wc)


# In[59]:


#Find the 30 most used words in spam 
spam_corpus = []
for msg in df[df['target']==1]['transformed'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[60]:


len(spam_corpus)


# In[81]:


from collections import Counter
import matplotlib.pyplot as plt
word_counts = Counter(spam_corpus).most_common(30)
df_word = pd.DataFrame(word_counts, columns=['word','frequent'])
sns.barplot(x='word', y='frequent', data=df_word)
plt.xticks(rotation='vertical')
plt.show()


# In[90]:


ham_corpus = []
for msg in df[df['target']==0]['transformed'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[91]:


word_count = Counter(ham_corpus).most_common(30)
df_words = pd.DataFrame(word_count, columns=['word', 'frequent'])
sns.barplot(x='word', y='frequent', data=df_words)
plt.xticks(rotation='vertical')
plt.show()


# ## Modelling

# In[130]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features = 3000)


# In[131]:


# X = cv.fit_transform(df['transformed']).toarray()
# change value of X for Tfidf
X = tfidf.fit_transform(df['transformed']).toarray()
y = df['target'].values


# In[132]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


# In[133]:


from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix 


# In[134]:


gnb = GaussianNB()
bnb = BernoulliNB()
mnb = MultinomialNB()


# In[135]:


gnb.fit(X_train, y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test, y_pred1))
print(precision_score(y_test, y_pred1))
print(confusion_matrix(y_test, y_pred1))


# In[136]:


mnb.fit(X_train, y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test, y_pred2))
print(precision_score(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))


# In[137]:


bnb.fit(X_train, y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test, y_pred3))
print(precision_score(y_test, y_pred3))
print(confusion_matrix(y_test, y_pred3))


# ## Creating pipeline

# In[139]:


import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('main.pkl','wb'))
pickle.dump(model, open('main.pkl', 'wb'))


# In[ ]:




