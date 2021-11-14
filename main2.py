import pandas as pd


df = pd.read_csv('Resume.csv')
df.head()
df.info()
df.isnull().sum()
df['Category'].value_counts()
df['Resume_str'][0]
