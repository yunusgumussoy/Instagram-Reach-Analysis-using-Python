# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 18:19:02 2023

@author: Yunus
"""
# source: https://statso.io/instagram-reach-analysis-case-study/
# https://thecleverprogrammer.com/2022/03/22/instagram-reach-analysis-using-python/

# pip install wordcloud


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
# pio.renderers.default='svg'
pio.renderers.default='browser'

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor

data = pd.read_csv("Instagram.csv", encoding = 'latin1')
print(data.head())

# whether this dataset contains any null values or not
data.isnull().sum()

# drop all these null values
data = data.dropna()

# understand the data type of all the columns:
data.info()

# Analyzing Instagram Reach
# distribution of impressions received from home
plt.figure(figsize=(10, 8))
plt.style.use('fivethirtyeight')
plt.title("Distribution of Impressions From Home")
# sns.distplot(data['From Home'])
sns.histplot(data['From Home'], kde = True)
plt.show()

# distribution of the impressions received from hashtags
plt.figure(figsize=(10, 8))
plt.title("Distribution of Impressions From Hashtags")
# sns.distplot(data['From Hashtags'])
sns.histplot(data['From Hashtags'], kde = True)
plt.show()

# distribution of impressions received from the explore section of Instagram
plt.figure(figsize=(10, 8))
plt.title("Distribution of Impressions From Explore")
# sns.distplot(data['From Explore'])
sns.histplot(data['From Explore'], kde = True)
plt.show()

# the percentage of impressions from various sources on Instagram
home = data["From Home"].sum()
hashtags = data["From Hashtags"].sum()
explore = data["From Explore"].sum()
other = data["From Other"].sum()

labels = ['From Home','From Hashtags','From Explore','Other']
values = [home, hashtags, explore, other]

fig = px.pie(data_frame = data, values=values, names=labels, title='Impressions on Instagram Posts From Various Sources')
fig.show ()


# Content Analysis
# creating a wordcloud of the caption column to look at the most used words in the caption of Instagram posts
text = " ".join(i for i in data.Caption)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.style.use('classic')
plt.figure( figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# creating a wordcloud of the hashtags column to look at the most used hashtags in Instagram posts
text = " ".join(i for i in data.Hashtags)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure( figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# Relationships Analysis
# the relationship between the number of likes and the number of impressions on Instagram posts
figure = px.scatter(data_frame = data, x="Impressions", y="Likes", size="Likes", trendline="ols", title = "Relationship Between Likes and Impressions")
figure.show()

# the relationship between the number of comments and the number of impressions on Instagram posts
figure = px.scatter(data_frame = data, x="Impressions", y="Comments", size="Comments", trendline="ols", title = "Relationship Between Comments and Total Impressions")
figure.show()

# the relationship between the number of shares and the number of impressions
figure = px.scatter(data_frame = data, x="Impressions", y="Shares", size="Shares", trendline="ols", title = "Relationship Between Shares and Total Impressions")
figure.show()

# the relationship between the number of saves and the number of impressions
figure = px.scatter(data_frame = data, x="Impressions", y="Saves", size="Saves", trendline="ols", title = "Relationship Between Post Saves and Total Impressions")
figure.show()

# the correlation of all the columns with the Impressions column
correlation = data.corr()
print(correlation["Impressions"].sort_values(ascending=False))

# Analyzing Conversion Rate
conversion_rate = (data["Follows"].sum() / data["Profile Visits"].sum()) * 100
print(conversion_rate)

# the relationship between the total profile visits and the number of followers gained from all profile visits
figure = px.scatter(data_frame = data, x="Profile Visits", y="Follows", size="Follows", trendline="ols", title = "Relationship Between Profile Visits and Followers Gained")
figure.show()

# Instagram Reach Prediction Model
x = np.array(data[['Likes', 'Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']])
y = np.array(data["Impressions"])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)

# Machine learning model to predict the reach of an Instagram post
model = PassiveAggressiveRegressor()
model.fit(xtrain, ytrain)
model.score(xtest, ytest)

# Predict the reach of an Instagram post by giving inputs to the machine learning model
# Features = [['Likes','Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']]
features = np.array([[282.0, 233.0, 4.0, 9.0, 165.0, 54.0]])
model.predict(features)
