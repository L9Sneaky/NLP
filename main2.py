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
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import plot_roc_curve
from collections import Counter
from sklearn import preprocessing
import re
from wordcloud import WordCloud
import pandas
import string
import nltk as nlp
from nltk.corpus import stopwords
import matplotlib.cm as cm
from matplotlib import rcParams
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
#%%
df = pd.read_csv('dataset.csv')
del df['Unnamed: 0']
df.info()

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


df["text"] = df["text"].apply(processPost)

#%%
df.head()
#%%
stopwords_list = stopwords.words('english')
df['text']=df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
#df["text"]=df["text"].apply(lambda x: [item for item in x if item not in stopwords_list])
#%%
df.head()
#%%
tokenizer = RegexpTokenizer(r'\w+')
df["text"] = df["text"].apply(tokenizer.tokenize)
#%%
df.head()
#%%
df['text'] = df['text'].apply(lambda x: ' '.join(map(str, x)))
df.head()
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
df.head()
#%%
X=df['text']
y=df['category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333, random_state=65)
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

#%%
cv2 = CountVectorizer(ngram_range=(1,2), binary=True, stop_words='english')

X_train_cv2 = cv2.fit_transform(X_train)
X_test_cv2  = cv2.transform(X_test)

#%%
tfidf1 = TfidfVectorizer(stop_words='english')

X_train_tfidf1 = tfidf1.fit_transform(X_train)
X_test_tfidf1  = tfidf1.transform(X_test)

#%%
tfidf2 = TfidfVectorizer(ngram_range=(1,2), binary=True, stop_words='english')

X_train_tfidf2 = tfidf2.fit_transform(X_train)
X_test_tfidf2  = tfidf2.transform(X_test)
#%%
#creating the objects
lr_cv1 = LogisticRegression()
svm_cv1 = SVC(kernel = 'linear', random_state = 0)
knn_cv1 = KNeighborsClassifier(n_neighbors=2)
rf_cv1 = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
cv1_dict = {0: 'Logistic Regression', 1: 'SVM',2:'KNN',3:'Random Forest'}
cv1_models=[lr_cv1,svm_cv1,knn_cv1,rf_cv1]
#%%
lr_cv1.fit(X_train_cv1, y_train)
y_pred_cv1=lr_cv1.predict(X_test_cv1)
cm1 = conf_matrix(y_test, y_pred_cv1)
#%%
svm_cv1.fit(X_train_cv1, y_train)
y_pred_cv1=svm_cv1.predict(X_test_cv1)
cm2 = conf_matrix(y_test, y_pred_cv1)
#%%
knn_cv1.fit(X_train_cv1, y_train)
y_pred_cv1=knn_cv1.predict(X_test_cv1)
cm3 = conf_matrix(y_test, y_pred_cv1)
#%%
rf_cv1.fit(X_train_cv1, y_train)
y_pred_cv1=rf_cv1.predict(X_test_cv1)
cm4 = conf_matrix(y_test, y_pred_cv1)
#%%
#creating the objects
lr_cv2 = LogisticRegression()
svm_cv2 = SVC(kernel = 'linear', random_state = 0)
knn_cv2 = KNeighborsClassifier(n_neighbors=2)
rf_cv2 = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
cv2_dict = {0: 'Logistic Regression', 1: 'SVM',2:'KNN',3:'Random Forest'}
cv2_models=[lr_cv2,svm_cv2,knn_cv2,rf_cv2]

#%%
lr_cv2.fit(X_train_cv2, y_train)
y_pred_cv2=lr_cv2.predict(X_test_cv2)
cm5 = conf_matrix(y_test, y_pred_cv2)
#%%
svm_cv2.fit(X_train_cv2, y_train)
y_pred_cv2=svm_cv2.predict(X_test_cv2)
cm6 = conf_matrix(y_test, y_pred_cv2)
#%%
knn_cv2.fit(X_train_cv2, y_train)
y_pred_cv2=knn_cv2.predict(X_test_cv2)
cm7 = conf_matrix(y_test, y_pred_cv2)
#%%
rf_cv2.fit(X_train_cv2, y_train)
y_pred_cv2=rf_cv2.predict(X_test_cv2)
cm8 = conf_matrix(y_test, y_pred_cv2)

#%%
lr_tfidf1 = LogisticRegression()
svm_tfidf1 = SVC(kernel = 'linear', random_state = 0)
knn_tfidf1 = KNeighborsClassifier(n_neighbors=2)
rf_tfidf1 = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
tfidf1_dict = {0: 'Logistic Regression', 1: 'SVM',2:'KNN',3:'Random Forest'}
tfidf1_models=[lr_tfidf1,svm_tfidf1,knn_tfidf1,rf_tfidf1]

#%%
lr_tfidf1.fit(X_train_tfidf1, y_train)
y_pred_tfidf1=lr_tfidf1.predict(X_test_tfidf1)
cm9 = conf_matrix(y_test, y_pred_tfidf1)
#%%
svm_tfidf1.fit(X_train_tfidf1, y_train)
y_pred_tfidf1=svm_tfidf1.predict(X_test_tfidf1)
cm10 = conf_matrix(y_test, y_pred_tfidf1)
#%%
knn_tfidf1.fit(X_train_tfidf1, y_train)
y_pred_tfidf1=knn_tfidf1.predict(X_test_tfidf1)
cm11 = conf_matrix(y_test, y_pred_tfidf1)
#%%
rf_tfidf1.fit(X_train_tfidf1, y_train)
y_pred_tfidf1=rf_tfidf1.predict(X_test_tfidf1)
cm12 = conf_matrix(y_test, y_pred_tfidf1)
#%%
lr_tfidf2 = LogisticRegression()
svm_tfidf2 = SVC(kernel = 'linear', random_state = 0)
knn_tfidf2 = KNeighborsClassifier(n_neighbors=2)
rf_tfidf2 = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
tfidf2_dict = {0: 'Logistic Regression', 1: 'SVM',2:'KNN',3:'Random Forest'}
tfidf2_models=[lr_tfidf2,svm_tfidf2,knn_tfidf2,rf_tfidf2]

#%%
lr_tfidf2.fit(X_train_tfidf2, y_train)
y_pred_tfidf2=lr_tfidf2.predict(X_test_tfidf2)
cm13 = conf_matrix(y_test, y_pred_tfidf2)
#%%
svm_tfidf2.fit(X_train_tfidf2, y_train)
y_pred_tfidf2=svm_tfidf2.predict(X_test_tfidf2)
cm14 = conf_matrix(y_test, y_pred_tfidf2)
#%%
knn_tfidf2.fit(X_train_tfidf2, y_train)
y_pred_tfidf2=knn_tfidf2.predict(X_test_tfidf2)
cm15 = conf_matrix(y_test, y_pred_tfidf2)
#%%
rf_tfidf2.fit(X_train_tfidf2, y_train)
y_pred_tfidf2=rf_tfidf2.predict(X_test_tfidf2)
cm16 = conf_matrix(y_test, y_pred_tfidf2)
#%%

#%%
kfold = KFold(n_splits=10) # k=10, split the data into 10 equal parts
xyz=[]
accuracy=[]
std=[]
classifiers=['Logistic Regression','SVM','KNN','Random Forest']
models=[LogisticRegression(),
        SVC(kernel = 'linear', random_state = 0),
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
cval(X_train_cv2,y_train)
cval(X_train_tfidf1,y_train)
cval(X_train_tfidf2,y_train)
models_dataframe=pd.DataFrame({'CV Mean':xyz,'Std':std},index=classifiers)
models_dataframe['CV Mean']
#%%

disp = plot_roc_curve(lr_cv1, X_test_cv1, y_test)
plot_roc_curve(svm_cv1, X_test_cv1, y_test, ax=disp.ax_)
plot_roc_curve(knn_cv1, X_test_cv1, y_test, ax=disp.ax_)
plot_roc_curve(rf_cv1, X_test_cv1, y_test, ax=disp.ax_)
plot_roc_curve(lr_cv2, X_test_cv2, y_test, ax=disp.ax_)
plot_roc_curve(svm_cv2, X_test_cv2, y_test, ax=disp.ax_)
plot_roc_curve(knn_cv2, X_test_cv2, y_test, ax=disp.ax_)
plot_roc_curve(rf_cv2, X_test_cv2, y_test, ax=disp.ax_)
plot_roc_curve(lr_tfidf1, X_test_tfidf1, y_test, ax=disp.ax_)
plot_roc_curve(svm_tfidf1, X_test_tfidf1, y_test, ax=disp.ax_)
plot_roc_curve(knn_tfidf1, X_test_tfidf1, y_test, ax=disp.ax_)
plot_roc_curve(rf_tfidf1, X_test_tfidf1, y_test, ax=disp.ax_)
plot_roc_curve(lr_tfidf2, X_test_tfidf2, y_test, ax=disp.ax_)
plot_roc_curve(svm_tfidf2, X_test_tfidf2, y_test, ax=disp.ax_)
plot_roc_curve(knn_tfidf2, X_test_tfidf2, y_test, ax=disp.ax_)
plot_roc_curve(rf_tfidf2, X_test_tfidf2, y_test, ax=disp.ax_)
#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%

#%%
results = pd.DataFrame(list(zip(cm1, cm2,cm3,cm4,cm5,cm6,cm8,cm8,cm9,cm10,cm11,cm12,cm13,cm14,cm15,cm16)))
results = results.set_index([['Accuracy', 'Precision', 'Recall', 'F1 Score']])
results.columns = ['LogRegcv1','SVMcv1','KNNcv1','RandForcv1', 'LogRegcv2','SVMcv2','KNNcv2','RandForcv2','LogRegtfidf1','SVMtfidf1','KNNtfidf1','RandFortfidf1','LogRegtfidf2','SVMtfidf2','KNNtfidf2','RandFortfidf2']
results.transpose()['Avg']=(results.transpose()['Accuracy']+results.transpose()['Precision']+results.transpose()['Recall']+results.transpose()['F1 Score'])/4
#%%
results.transpose()

#%%

#%%

#%%

#%%
