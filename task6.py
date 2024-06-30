import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("disney_plus_titles.csv")

#Q1 - What are the 3 most common genres on Disney Plus?
gen = df['listed_in'].str.split(', ').explode()
topgen = gen.value_counts().head(3)
plt.figure(figsize=(10, 6))
sns.barplot(x=topgen.index, y=topgen.values)
plt.title('3 Most Common Genres on Disney Plus')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.show()

#Q2 - How has the number of new titles added per year changed over the past 5 years?
df['date_added'] = pd.to_datetime(df['date_added'])
df['year_added'] = df['date_added'].dt.year
lf = df[df['year_added'] >= (df['year_added'].max() - 5)]
tf = lf['year_added'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
sns.lineplot(x=tf.index, y=tf.values, marker='o')
plt.title('Number of New Titles Added Per Year(Last 5 Years)')
plt.xlabel('Year')
plt.ylabel('Number of New Titles')
plt.show()

#Q3 - Which country produces the most content, and what is its most popular genre?
cc = df['country'].value_counts()
tc = cc.idxmax()
toptc = df[df['country'] == tc]['listed_in'].str.split(', ').explode().value_counts().head(8)
plt.figure(figsize=(10, 6))
sns.barplot(x=toptc.index, y=toptc.values)
plt.title(f'Most Popular Genres in {tc}')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.show()

#Q4 -  What is the average duration of movies and TV shows on Disney Plus?
df['numeric_duration'] = df['duration'].str.extract('(\d+)').astype(float)
ad = df.groupby('type')['numeric_duration'].mean()
plt.figure(figsize=(10, 6))
sns.barplot(x=ad.index, y=ad.values)
plt.title('Average Duration of Movies and TV Shows')
plt.xlabel('Type')
plt.ylabel('Average Duration (minutes)')
plt.show()

#Q5 - What is the correlation between numerical features like Duration and Year?
df['numeric_duration'] = df['duration'].str.extract('(\d+)').astype(float)
nf = df[['numeric_duration', 'release_year']]
cm = nf.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()
