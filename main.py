import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from collections import defaultdict,Counter
from rich.progress import track
import re
df1 = pd.read_csv('Corona_NLP_test.csv')
df2 = pd.read_csv('Corona_NLP_train.csv',encoding='ISO-8859-1')


df =pd.concat([df1, df2], ignore_index=True)
df.head()
df.info()
df.drop_duplicates()
print(" Shape of dataframe after dropping duplicates: ", df.shape)
#%%
df.isnull().sum().sort_values(ascending=False)
df = df.dropna()
df.isnull().sum().sort_values(ascending=False)
df.shape

#%%
target=df['Sentiment']

df['Sentiment'].value_counts(normalize= True)
#%%
class_df = df.groupby('Sentiment').count()['OriginalTweet'].reset_index().sort_values(by='OriginalTweet',ascending=False)
class_df.style.background_gradient(cmap='winter')
#%%
percent_class=class_df.OriginalTweet
labels= class_df.Sentiment

colors = ['#17C37B','#F92969','#FACA0C']
my_pie,_,_ = plt.pie(percent_class,radius = 1.2,labels=labels,colors=colors,autopct="%.1f%%")

plt.setp(my_pie, width=0.6, edgecolor='white')
plt.show()

#%%

fig,(ax4,ax1,ax3,ax2,ax5)=plt.subplots(1,5,figsize=(25,5))

tweet_len=df[df['Sentiment']=="Positive"]['OriginalTweet'].str.len()
ax1.hist(tweet_len,color='#17C37B')
ax1.set_title('Positive Sentiments')

tweet_len=df[df['Sentiment']=="Negative"]['OriginalTweet'].str.len()
ax2.hist(tweet_len,color='#F92969')
ax2.set_title('Negative Sentiments')

tweet_len=df[df['Sentiment']=="Neutral"]['OriginalTweet'].str.len()
ax3.hist(tweet_len,color='#FACA0C')
ax3.set_title('Neutral Sentiments')

tweet_len=df[df['Sentiment']=="Extremely Positive"]['OriginalTweet'].str.len()
ax4.hist(tweet_len,color='#A6CA0C')
ax4.set_title('Extremely Positive Sentiments')

tweet_len=df[df['Sentiment']=="Extremely Negative"]['OriginalTweet'].str.len()
ax5.hist(tweet_len,color='#85CA0C')
ax5.set_title('Extremely Negative Sentiments')

fig.suptitle('Characters in tweets')
plt.show()
#%%
fig,(ax4,ax1,ax3,ax2,ax5)=plt.subplots(1,5,figsize=(25,5))

tweet_len=df[df['Sentiment']=="Positive"]['OriginalTweet'].str.split().map(lambda x: len(x))
ax1.hist(tweet_len,color='#17C37B')
ax1.set_title('Positive Sentiments')


tweet_len=df[df['Sentiment']=="Negative"]['OriginalTweet'].str.split().map(lambda x: len(x))
ax2.hist(tweet_len,color='#F92969')
ax2.set_title('Negative Sentiments')

tweet_len=df[df['Sentiment']=="Neutral"]['OriginalTweet'].str.split().map(lambda x: len(x))
ax3.hist(tweet_len,color='#FACA0C')
ax3.set_title('Neutral Sentiments')


tweet_len=df[df['Sentiment']=="Extremely Positive"]['OriginalTweet'].str.split().map(lambda x: len(x))
ax4.hist(tweet_len,color='#A6CA0C')
ax4.set_title('Extremely Positive Sentiments')

tweet_len=df[df['Sentiment']=="Extremely Negative"]['OriginalTweet'].str.split().map(lambda x: len(x))
ax5.hist(tweet_len,color='#85CA0C')
ax5.set_title('Extremely Negative Sentiments')

fig.suptitle('Words in a tweet')
plt.show()
#%%

fig,(ax4,ax1,ax3,ax2,ax5)=plt.subplots(1,5,figsize=(25,5))

word=df[df['Sentiment']=="Positive"]['OriginalTweet'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax1,color='#17C37B')
ax1.set_title('Positive')


word=df[df['Sentiment']=="Negative"]['OriginalTweet'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax2,color='#F92969')
ax2.set_title('Negative')

word=df[df['Sentiment']=="Neutral"]['OriginalTweet'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax3,color='#FACA0C')
ax3.set_title('Neutral')

word=df[df['Sentiment']=="Extremely Positive"]['OriginalTweet'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax4,color='#A6CA0C')
ax4.set_title('Extremely Positive Sentiments')

word=df[df['Sentiment']=="Extremely Negative"]['OriginalTweet'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax5,color='#85CA0C')
ax5.set_title('Extremely Negative Sentiments')


fig.suptitle('Average word length in each tweet')


#%%

def create_corpus(target):
    corpus=[]
    for x in df[df['Sentiment'] == target ]['OriginalTweet'].str.split():
        for i in x:
            corpus.append(i)
    return corpus

n = np.array(create_corpus('Extremely Positive'))
n
#%%
df['Sentiment'].value_counts().keys()
stop = np.array([])
for i in range(len(df['Sentiment'].value_counts())):
    temp=create_corpus(df['Sentiment'].value_counts().keys()[i])
    stop = np.append(stop,temp)

#%%

comment_words = ''
stopwords = set(stop)

for val in stop:
    # typecaste each val to string
    val = str(val)
    # split the value
    tokens = val.split()
    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()

    comment_words += " ".join(tokens)+" "

wordcloud = WordCloud(width = 1000, height = 1000,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)

# plot the WordCloud image
plt.figure(figsize = (8, 8), facecolor = "white")
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)

plt.show()


#%%
counter=Counter(stop)
most=counter.most_common()
x=[]
y=[]
for word,count in most[:40]:
        x.append(word)
        y.append(count)
plt.figure(figsize = (8, 8))
sns.barplot(x=y,y=x)
#%%
#Remove Urls and HTML links
def remove_urls(OriginalTweet):
    url_remove = re.compile(r'https?://\S+|www\.\S+')
    return url_remove.sub(r'', OriginalTweet)
df['OriginalTweet_new']=df['OriginalTweet'].apply(lambda x:remove_urls(x))

def remove_html(OriginalTweet):
    html=re.compile(r'<.*?>')
    return html.sub(r'',OriginalTweet)
df['OriginalTweet']=df['OriginalTweet_new'].apply(lambda x:remove_html(x))

# Lower casing
def lower(OriginalTweet):
    low_OriginalTweet= OriginalTweet.lower()
    return low_OriginalTweet
df['OriginalTweet_new']=df['OriginalTweet'].apply(lambda x:lower(x))

# Number removal
def remove_num(OriginalTweet):
    remove= re.sub(r'\d+', '', OriginalTweet)
    return remove
df['OriginalTweet']=df['OriginalTweet_new'].apply(lambda x:remove_num(x))

#Remove stopwords & Punctuations
from nltk.corpus import stopwords
", ".join(stopwords.words('english'))
STOPWORDS = set(stopwords.words('english'))

def punct_remove(OriginalTweet):
    punct = re.sub(r"[^\w\s\d]","", OriginalTweet)
    return punct
df['OriginalTweet_new']=df['OriginalTweet'].apply(lambda x:punct_remove(x))

def remove_stopwords(OriginalTweet):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(OriginalTweet).split() if word not in STOPWORDS])
df['OriginalTweet']=df['OriginalTweet_new'].apply(lambda x:remove_stopwords(x))

#Remove mentions and hashtags
def remove_mention(x):
    OriginalTweet=re.sub(r'@\w+','',x)
    return OriginalTweet
df['OriginalTweet_new']=df['OriginalTweet'].apply(lambda x:remove_mention(x))

def remove_hash(x):
    OriginalTweet=re.sub(r'#\w+','',x)
    return OriginalTweet
df['OriginalTweet']=df['OriginalTweet_new'].apply(lambda x:remove_hash(x))

#Remove extra white space left while removing stuff
def remove_space(OriginalTweet):
    space_remove = re.sub(r"\s+"," ",OriginalTweet).strip()
    return space_remove
df['OriginalTweet_new']=df['OriginalTweet'].apply(lambda x:remove_space(x))

df = df.drop(columns=['OriginalTweet_new'])
#%%
df.shape
#%%
fig,(ax4,ax1,ax3,ax2,ax5)= plt.subplots(1, 5, figsize=[90, 15])

df_pos = df[df["Sentiment"]=="Positive"]
df_neg = df[df["Sentiment"]=="Negative"]
df_neu = df[df["Sentiment"]=="Neutral"]
df_epos = df[df["Sentiment"]=="Extremely Positive"]
df_eneg = df[df["Sentiment"]=="Extremely Negative"]


comment_words = ''
stopwords = set(stop)

for val in df_pos.OriginalTweet:

    # typecaste each val to string
    val = str(val)

    # split the value
    tokens = val.split()

    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()

    comment_words += " ".join(tokens)+" "


wordcloud1 = WordCloud(width = 800, height = 800,
                background_color ='white',
                colormap="Greens",
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)

ax1.imshow(wordcloud1)
ax1.axis('off')
ax1.set_title('Positive Sentiment',fontsize=35);

##

comment_words = ''

for val in df_neg.OriginalTweet:
    # typecaste each val to string
    val = str(val)
    # split the value
    tokens = val.split()
    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
    comment_words += " ".join(tokens)+" "


wordcloud2 = WordCloud(width = 800, height = 800,
                background_color ='white',
                colormap="Reds",
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)
ax2.imshow(wordcloud2)
ax2.axis('off')
ax2.set_title('Negative Sentiment',fontsize=35);

##

comment_words = ''
for val in df_neu.OriginalTweet:
    # typecaste each val to string
    val = str(val)
    # split the value
    tokens = val.split()
    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
    comment_words += " ".join(tokens)+" "

wordcloud3 = WordCloud(width = 800, height = 800,
                background_color ='white',
                colormap="Greys",
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)
ax3.imshow(wordcloud3)
ax3.axis('off')
ax3.set_title('Neutal Sentiment',fontsize=35);

##

comment_words = ''
for val in df_epos.OriginalTweet:
    # typecaste each val to string
    val = str(val)
    # split the value
    tokens = val.split()
    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
    comment_words += " ".join(tokens)+" "

wordcloud3 = WordCloud(width = 800, height = 800,
                background_color ='white',
                colormap="Greys",
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)
ax4.imshow(wordcloud3)
ax4.axis('off')
ax4.set_title('Extremely Positive',fontsize=35);

##

comment_words = ''
for val in df_eneg.OriginalTweet:
    # typecaste each val to string
    val = str(val)
    # split the value
    tokens = val.split()
    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
    comment_words += " ".join(tokens)+" "

wordcloud3 = WordCloud(width = 800, height = 800,
                background_color ='white',
                colormap="Greys",
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)
ax5.imshow(wordcloud3)
ax5.axis('off')
ax5.set_title('Extremely Negative',fontsize=35);
#%%
