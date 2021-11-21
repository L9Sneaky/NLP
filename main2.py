import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import plot_roc_curve
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA1
from collections import Counter
from sklearn import preprocessing
import re
from wordcloud import WordCloud
import string
import nltk as nlp
from nltk.corpus import stopwords
import matplotlib.cm as cm
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import re,string,unicodedata
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk import pos_tag
from nltk.corpus import wordnet
import warnings
warnings.filterwarnings("ignore")

#%%
# %% codecell
df1 = pd.read_csv('True.csv')
df2 = pd.read_csv('Fake.csv')
df1['category']=1
df2['category']=0
df = pd.concat([df1, df2], axis=0,ignore_index=True)


# %% codecell
sns.set_style("darkgrid")
sns.countplot(df.category)
#%%
df.head()
# %% codecell
df.isna().sum() # Checking for nan Values
# %% codecell
df.title.count()
# %% codecell
df.subject.value_counts()
#%%
plt.figure(figsize = (12,8))
sns.set(style = "whitegrid",font_scale = 1.2)
chart = sns.countplot(x = "subject", hue = "category" , data = df)
chart.set_xticklabels(chart.get_xticklabels(),rotation=90)
#%%
df['text'] = df['text'] + " " + df['title']
del df['title']
del df['subject']
del df['date']

stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)
#%%

# %% codecell
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)
# Removing URL's
def remove_between_square_brackets(text):
    return re.sub(r'http\S+', '', text)
#Removing the stopwords from text
def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            final_text.append(i.strip())
    return " ".join(final_text)
#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_stopwords(text)
    return text
#Apply function on review column
df['text']=df['text'].apply(denoise_text)

# %% codecell
plt.figure(figsize = (20,20)) # Text that is not Fake
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(df[df.category == 1].text))
plt.imshow(wc , interpolation = 'bilinear')
plt.title('WORDCLOUD FOR REAL TEXT')

# %% codecell
plt.figure(figsize = (20,20)) # Text that is Fake
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(df[df.category == 0].text))
plt.imshow(wc , interpolation = 'bilinear')
plt.title('WORDCLOUD FOR FAKE TEXT')

# %% codecell
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,8))
text_len=df[df['category']==1]['text'].str.len()
ax1.hist(text_len,color='red')
ax1.set_title('Original text')
text_len=df[df['category']==0]['text'].str.len()
ax2.hist(text_len,color='green')
ax2.set_title('Fake text')
fig.suptitle('Characters in texts')
plt.show()

# %% codecell
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,8))
text_len=df[df['category']==1]['text'].str.split().map(lambda x: len(x))
ax1.hist(text_len,color='red')
ax1.set_title('Original text')
text_len=df[df['category']==0]['text'].str.split().map(lambda x: len(x))
ax2.hist(text_len,color='green')
ax2.set_title('Fake text')
fig.suptitle('Words in texts')
plt.show()

# %% codecell
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(20,10))
word=df[df['category']==1]['text'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax1,color='red')
ax1.set_title('Original text')
word=df[df['category']==0]['text'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax2,color='green')
ax2.set_title('Fake text')
fig.suptitle('Average word length in each text')
# %% codecell
def get_corpus(text):
    words = []
    for i in text:
        for j in i.split():
            words.append(j.strip())
    return words
corpus = get_corpus(df.text)
corpus[:5]
# %% codecell
from collections import Counter
counter = Counter(corpus)
most_common = counter.most_common(10)
most_common = dict(most_common)
most_common
# %% codecell
from sklearn.feature_extraction.text import CountVectorizer
def get_top_text_ngrams(corpus, n, g):
    vec = CountVectorizer(ngram_range=(g, g)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
#%%
plt.figure(figsize = (16,9))
most_common_uni = get_top_text_ngrams(df.text,10,1)
most_common_uni = dict(most_common_uni)
sns.barplot(x=list(most_common_uni.values()),y=list(most_common_uni.keys()))
#%%
plt.figure(figsize = (16,9))
most_common_bi = get_top_text_ngrams(df.text,10,2)
most_common_bi = dict(most_common_bi)
sns.barplot(x=list(most_common_bi.values()),y=list(most_common_bi.keys()))
#%%
plt.figure(figsize = (16,9))
most_common_tri = get_top_text_ngrams(df.text,10,3)
most_common_tri = dict(most_common_tri)
sns.barplot(x=list(most_common_tri.values()),y=list(most_common_tri.keys()))
#%%
df1 = pd.read_csv('True.csv')
df2 = pd.read_csv('Fake.csv')
df1['category']=1
df2['category']=0
df = pd.concat([df1, df2], axis=0,ignore_index=True)
df.info()
#%%
target = ['Fake', 'True',]
#%%
df.head()
#%%
sns.set_style("darkgrid")
sns.countplot(df.category)
#%%
df['text'] = df['text'] + " " + df['title']
del df['title']
del df['subject']
del df['date']

stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)
stop
#%%

df.head()
#%%

for letter in '1234567890.(/ÀÈÌÒÙàèìòùÁÉÍÓÚÝáéíóúýÂÊÎÔÛâêîôûÃÑÕãñõÄËÏÖÜŸäëïöüÿ':
    df["text"] = df["text"].str.replace(letter,'')

english_punctuations = string.punctuation
punctuations_list = english_punctuations + english_punctuations

def remove_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)

def remove_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1', text)

def processPost(text:str):
    text = re.sub('@[^\s]+', ' ', text)
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',text)

    text = re.sub(r'#([^\s]+)', r'\1', text)

    text= remove_punctuations(text)
    text=remove_repeating_char(text)
    text=text.lower()
    return text

df=df.sample(frac=0.9, replace=True, random_state=1)
df["text"] = df["text"].apply(processPost)

#%%
df.head()
#%%
stopwords_list = stopwords.words('english')
df['text']=df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
#df["text"]=df["text"].apply(lambda x: [item for item in x if item not in stopwords_list])
#%%
df['text'][0]
#%%
#tokenizer = RegexpTokenizer(r'\w+')
#df["text"] = df["text"].apply(tokenizer.tokenize)
#%%
#df.head()
#%%
#df['text'] = df['text'].apply(lambda x: ' '.join(map(str, x)))
#df.head()
#%%
text = df.iloc[0][0]
text
w_tokenizer = nlp.tokenize.WhitespaceTokenizer()
lemmatizer = nlp.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

df['text'] = df['text'].apply(lemmatize_text)
df['text'] = df['text'].apply(lambda x: ' '.join(map(str, x)))
#%%
df.head(10)
#%%

X=df['text']
y=df['category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45, random_state=65)
#%%
def conf_matrix(actual, predicted):
    cm = confusion_matrix(actual, predicted)
    sns.heatmap(cm, xticklabels=['predicted_negative', 'predicted_positive'],
                yticklabels=['actual_negative', 'actual_positive'], annot=True,
                fmt='d', annot_kws={'fontsize':20}, cmap="YlGnBu");

    true_neg, false_pos = cm[0]
    false_neg, true_pos = cm[1]

    accuracy = round((true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg),3)
    precision = round((true_pos) / (true_pos + false_pos),3)
    recall = round((true_pos) / (true_pos + false_neg),3)
    f1 = round(2 * (precision * recall) / (precision + recall),3)

    cm_results = [accuracy, precision, recall, f1]
    return cm_results


#%%
cv1 = CountVectorizer(stop_words='english')
X_train_cv1 = cv1.fit_transform(X_train)
X_test_cv1  = cv1.transform(X_test)

#%%
#pd.DataFrame(X_train_cv1.toarray(), columns=cv1.get_feature_names())
pd.DataFrame(X_train_cv1.toarray(), columns=cv1.get_feature_names()).head()

#%%
tfidf1 = TfidfVectorizer(stop_words='english')

X_train_tfidf1 = tfidf1.fit_transform(X_train)
X_test_tfidf1  = tfidf1.transform(X_test)

#%%
#creating the objects
lr_cv1 = LogisticRegression()
knn_cv1 = KNeighborsClassifier(n_neighbors=2)
rf_cv1 = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
cv1_dict = {0: 'Logistic Regression', 1:'KNN', 2:'Random Forest'}
cv1_models=[lr_cv1,knn_cv1,rf_cv1]
#%%
lr_cv1.fit(X_train_cv1, y_train)
y_pred_cv1=lr_cv1.predict(X_test_cv1)
cm1 = conf_matrix(y_test, y_pred_cv1)

#%%
knn_cv1.fit(X_train_cv1, y_train)
y_pred_cv1=knn_cv1.predict(X_test_cv1)
cm2 = conf_matrix(y_test, y_pred_cv1)
#%%
rf_cv1.fit(X_train_cv1, y_train)
y_pred_cv1=rf_cv1.predict(X_test_cv1)
cm3 = conf_matrix(y_test, y_pred_cv1)

#%%
lr_tfidf1 = LogisticRegression()
knn_tfidf1 = KNeighborsClassifier(n_neighbors=2)
rf_tfidf1 = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
tfidf1_dict = {0: 'Logistic Regression', 1:'KNN', 2:'Random Forest'}
tfidf1_models=[lr_tfidf1,knn_tfidf1,rf_tfidf1]

#%%
lr_tfidf1.fit(X_train_tfidf1, y_train)
y_pred_tfidf1=lr_tfidf1.predict(X_test_tfidf1)
cm4 = conf_matrix(y_test, y_pred_tfidf1)

#%%
knn_tfidf1.fit(X_train_tfidf1, y_train)
y_pred_tfidf1=knn_tfidf1.predict(X_test_tfidf1)
cm5 = conf_matrix(y_test, y_pred_tfidf1)
#%%
rf_tfidf1.fit(X_train_tfidf1, y_train)
y_pred_tfidf1=rf_tfidf1.predict(X_test_tfidf1)
cm6 = conf_matrix(y_test, y_pred_tfidf1)
#%%
kfold = KFold(n_splits=10) # k=10, split the data into 10 equal parts
xyz=[]
accuracy=[]
std=[]
classifiers=['Logistic Regression','KNN','Random Forest']
models=[LogisticRegression(),
        KNeighborsClassifier(n_neighbors=2),
        RandomForestClassifier(n_estimators = 100, criterion = 'entropy')]
def cval(X,y):
    for i in models:
        model = i
        cv_result = cross_val_score(model,X,y, cv = kfold,scoring = "accuracy")
        cv_result = cv_result
        xyz.append(cv_result.mean())
        std.append(cv_result.std())
        accuracy.append(cv_result)

cval(X_train_cv1,y_train)
cval(X_train_tfidf1,y_train)

#%%
#models_dataframe['CV Mean']
classifiers=['LogRegcv1','KNNcv1','RandForcv1','LogRegtfidf1','KNNtfidf1','RandFortfidf1']
models_dataframe=pd.DataFrame({'CV Mean':xyz},index=classifiers)
models_dataframe
#%%

disp = plot_roc_curve(lr_cv1, X_test_cv1, y_test)
plot_roc_curve(knn_cv1, X_test_cv1, y_test, ax=disp.ax_)
plot_roc_curve(rf_cv1, X_test_cv1, y_test, ax=disp.ax_)
plot_roc_curve(lr_tfidf1, X_test_tfidf1, y_test, ax=disp.ax_)
plot_roc_curve(knn_tfidf1, X_test_tfidf1, y_test, ax=disp.ax_)
plot_roc_curve(rf_tfidf1, X_test_tfidf1, y_test, ax=disp.ax_)
#%%
results = pd.DataFrame(list(zip(cm1,cm2,cm3,cm4,cm5,cm6)))
results = results.set_index([['Accuracy', 'Precision', 'Recall', 'F1 Score']])
results.columns = ['LogRegcv1','KNNcv1','RandForcv1','LogRegtfidf1','KNNtfidf1','RandFortfidf1']
results.transpose()['Avg']=(results.transpose()['Accuracy']+results.transpose()['Precision']+results.transpose()['Recall']+results.transpose()['F1 Score'])/4
#%%
results.transpose().sort_values(['Accuracy','F1 Score','Precision','Recall'],ascending=False)

#%%
pd.concat([models_dataframe,results.transpose()],axis =1)
#%%
def predict_text(text:str):
    text = [processPost(text)]
    text = tfidf1.transform(text)
    print(target[knn_tfidf1.predict(text)[0]])
#%%
predict_text('Trump hates obama')
predict_text('Trump is poor')
|#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%
