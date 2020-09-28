# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 14:33:53 2020

@author: aditi
"""

# Importing Librariers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Importing dataset
df = pd.read_csv('Mall_Customers.csv')
df.head()


## Data Exploration
df.info()  ## TOTAL 200 VALUES
df.isnull().sum()  ## NO NULL VALUES
df.duplicated().sum()   ## NO DUPLICATES
np.where(df == " ")  ## NO BLANK STRINGS
df.Genre.value_counts() ## CHECKING VALUE COUNTS OF GENDER
df.describe()

## CHECKING FOR OUTLIERS
for col in df.columns:
    if df[col].dtype != object:
        sns.boxplot(df[col])
        plt.show()
## ONE OUTLIER IN ANNUAL INCOME


## CALCULATING RANGE OF UPPER FENCE AND LOWER FENCE FOR OUTLIER TREATMENT
q1, q3 = np.percentile(df['Annual Income (k$)'], [25,75])
iqr = q3-q1
lower_fence = q1-(1.5 * iqr)
upper_fence = q3+(1.5 * iqr)
## REPLACING THE OUTLIERS
df['Annual Income (k$)'] = df['Annual Income (k$)'].apply(lambda x: upper_fence if x > upper_fence else lower_fence if x< lower_fence else x)
sns.boxplot(df['Annual Income (k$)'])  ## OUTLIER REMOVED


sns.heatmap(df.corr(), annot = True, fmt = '0.2f', cmap="YlGnBu")
## Age and spending score have a higher negative correlation compared to 
## other variables
## seems logical


## Building a model with variables gender, age, income and spending score
dataframe = df.iloc[:, -4:]
dataframe.columns

##gender will have to be encoded before making the model
dataframe = pd.get_dummies(dataframe, columns = ['Genre'], drop_first= True)
dataframe.head()

sns.heatmap(dataframe.corr(), annot = True, fmt ='0.2f', cmap ='YlGnBu')
## spending score and age are related (negative corr)

# Importing K-means
from sklearn.cluster import KMeans

distance = []
k = range(3,11)

for value in k:
    km_model = KMeans(n_clusters = value, random_state = 42)
    km_model.fit(dataframe)
    distance.append(km_model.inertia_)
    
plt.plot(k, distance)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

## it is giving us 4 sharp points, 4,5 and 6 and 8
## so we cna check for 4,5,6 and 8 clusters

##Fitting k means to the datset first using 4
km_model = KMeans(n_clusters = 4, random_state = 42)
y_pred_1 = km_model.fit_predict(dataframe)
y_pred_1

dataframe['4 clusters result'] = y_pred_1
dataframe['4 clusters result'].unique()
dataframe['4 clusters result'].value_counts()  ## CLUSTER 2 HAS MAX VALUES


dataframe.groupby('4 clusters result')['Annual Income (k$)'].mean()
## CLUSTER 1 AND 3 HAVE THE SAME MEAN
## DOESN'T SEEM A GOOD BIFURCATION

dataframe.groupby('4 clusters result')['Spending Score (1-100)'].mean()

dataframe.groupby('4 clusters result')['Age'].mean()

dataframe.groupby('4 clusters result')['Genre_Male'].value_counts()


dataframe.groupby('4 clusters result')['Spending Score (1-100)'].mean().plot(kind = 'bar')
plt.xlabel('Category')
plt.ylabel('Average Spending Score (1-100)')
## Category 0 and 1 have the highest spending score


dataframe.groupby('4 clusters result')['Annual Income (k$)'].mean().plot(kind = 'bar')
plt.xlabel('Category')
plt.ylabel('Annual Income (k$)')
## Categories 1 and 3 are having the highest incomes
## Category 1 also has the highest spending score
## This segment can be considered as high income, high spending

dataframe.groupby('4 clusters result')['Age'].mean().plot(kind = 'bar')
## the average age of category 2 is around 45-50 
## average age of category 3 around 40
## category 3 are people in the age group of 40's with highest incomes
## and spending scores

dataframe.groupby('4 clusters result')['Genre_Male'].value_counts().plot(kind = 'bar')
## category 3 has larger number of males
## category 0 has equal numbers
## category 1 has larger number of males
## category 2 has learger numner of females and highest number of people


## MAKING ANOTHER MODEL WITH K = 5
km_model_2 = KMeans(n_clusters = 5, random_state = 42)
y_pred_2 = km_model_2.fit_predict(dataframe)
y_pred_2

dataframe['5 clusters result'] = y_pred_2
dataframe['5 clusters result'].unique()
dataframe['5 clusters result'].value_counts()  ## CLUSTER 1 HAS MAX VALUES


dataframe.groupby('5 clusters result')['Annual Income (k$)'].mean()
## CLUSTER 0 AND 3 HAVE NEARLY THE SAME MEAN

dataframe.groupby('5 clusters result')['Spending Score (1-100)'].mean()

dataframe.groupby('5 clusters result')['Age'].mean()

dataframe.groupby('5 clusters result')['Genre_Male'].value_counts()

dataframe.groupby('5 clusters result')['Spending Score (1-100)'].mean().plot(kind = 'bar')
plt.xlabel('Category')
plt.ylabel('Average Spending Score (1-100)')
## Categories 2 and 3 have the highest spending score


dataframe.groupby('5 clusters result')['Annual Income (k$)'].mean().plot(kind = 'bar')
plt.xlabel('Category')
plt.ylabel('Annual Income (k$)')
## Categories 0 and 3 are having the highest incomes
## Category 3 also has the highest spending score
## This segment can be considered as high income, high spending

dataframe.groupby('5 clusters result')['Age'].mean().plot(kind = 'bar')
## the average age of category 0 is around 40
## average age of category 1 around 45
## category 3 are people in the age group of 30-35 with highest incomes
## and spending scores
## category 2 lies in the average age range of 25
## category 4 has max average age, around 50


dataframe.groupby('5 clusters result')['Genre_Male'].value_counts().plot(kind = 'bar')
## category 1 has larger number of females and highest number of people


## MAKING ANOTHER MODEL WITH K = 6
km_model_3 = KMeans(n_clusters = 6, random_state = 42)
y_pred_3 = km_model_3.fit_predict(dataframe)
y_pred_3

dataframe['6 clusters result'] = y_pred_3
dataframe['6 clusters result'].unique()
dataframe['6 clusters result'].value_counts()  ## CLUSTER 2 HAS MAX VALUES


dataframe.groupby('6 clusters result')['Annual Income (k$)'].mean()
## CLUSTER 4 AND 5 HAVE THE SAME MEAN

dataframe.groupby('6 clusters result')['Spending Score (1-100)'].mean()
## CLUSTERS 1 AND 2 HAVE THE SAME MEAN

dataframe.groupby('6 clusters result')['Age'].mean()

dataframe.groupby('6 clusters result')['Genre_Male'].value_counts()
## Categories 2 and 3 have the highest spending score


dataframe.groupby('6 clusters result')['Spending Score (1-100)'].mean().plot(kind = 'bar')
plt.xlabel('Category')
plt.ylabel('Spending Score (1-100)')
## Categories 3 and 5 have highest spending scores

dataframe.groupby('6 clusters result')['Annual Income (k$)'].mean().plot(kind = 'bar')
plt.xlabel('Category')
plt.ylabel('Annual Income (k$)')
## Categories 0 and 3 are having the highest incomes
## Category 3 also has the highest spending score
## This segment can be considered as high income, high spending

dataframe.groupby('6 clusters result')['Age'].mean().plot(kind = 'bar')
## Catergory 2 has max age
## Category 1 and 5 have almost same age

dataframe.groupby('6 clusters result')['Genre_Male'].value_counts().plot(kind = 'bar')
## category 1 has larger number of females and highest number of people


## MAKING ANOTHER MODEL WITH K = 8
km_model_4 = KMeans(n_clusters = 8, random_state = 42)
y_pred_4 = km_model_4.fit_predict(dataframe)
y_pred_4

dataframe['8 clusters result'] = y_pred_4
dataframe['8 clusters result'].unique()
dataframe['8 clusters result'].value_counts()  ## CLUSTER 1 HAS MAX VALUES


dataframe.groupby('8 clusters result')['Annual Income (k$)'].mean()

dataframe.groupby('8 clusters result')['Spending Score (1-100)'].mean()

dataframe.groupby('8 clusters result')['Age'].mean()

dataframe.groupby('8 clusters result')['Genre_Male'].value_counts()


dataframe.groupby('8 clusters result')['Spending Score (1-100)'].mean().plot(kind = 'bar')
plt.xlabel('Category')
plt.ylabel('Spending Score (1-100)')


dataframe.groupby('8 clusters result')['Annual Income (k$)'].mean().plot(kind = 'bar')
plt.xlabel('Category')
plt.ylabel('Annual Income (k$)')

dataframe.groupby('8 clusters result')['Age'].mean().plot(kind = 'bar')

dataframe.groupby('8 clusters result')['Genre_Male'].value_counts().plot(kind = 'bar')

## Only 5 clusters seem to be most reasonable because there is a 
## considerable diffrence between the means of various clusters







## MAKING A MODEL WITH ONLY 2 VARIABLES
## SPENDING SCORE AND ANNUAL INCOME

## customer id should be dropped before final model making
df_new = df.iloc[:, [3,4]].values

# Importing K-means
from sklearn.cluster import KMeans


for i in k:
    kmeans = KMeans(n_clusters = i, random_state = 42)
    kmeans.fit(df_new)
    distance.append(kmeans.inertia_)
    
plt.plot(k, distance)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


##Fitting k means to the datset
kmeans = KMeans(n_clusters = 5, random_state = 42)
y_pred = kmeans.fit_predict(df_new)

df['result'] = y_pred
df.head(15)
df['result'].value_counts()
df['result'].unique()

df['Annual Income (k$)'].max()
df['Annual Income (k$)'].min()
df.groupby('result')['Annual Income (k$)'].mean()
df.groupby('result')['Spending Score (1-100)'].mean()


df.groupby('result')['Spending Score (1-100)'].mean().plot(kind = 'bar')
df.groupby('result')['Annual Income (k$)'].mean().plot(kind = 'bar')



cluster_0_points = df[df['result']==0].index
cluster_1_points = df[df['result']==1].index
cluster_2_points = df[df['result']==2].index
cluster_3_points = df[df['result']==3].index
cluster_4_points = df[df['result']==4].index

kmeans.cluster_centers_    ## centroid values

## Visualising the clusters
plt.scatter(df_new[cluster_0_points, 0],
            df_new[cluster_0_points, 1],
            s =100,
            c = 'cyan',
            label = 'category_1')

plt.scatter(df_new[cluster_1_points, 0],
            df_new[cluster_1_points, 1],
            s =100,
            c = 'red',
            label = 'category_2')


plt.scatter(df_new[cluster_2_points, 0],
            df_new[cluster_2_points, 1],
            s =100,
            c = 'blue',
            label = 'category_3')

plt.scatter(df_new[cluster_3_points, 0],
            df_new[cluster_3_points, 1],
            s =100,
            c = 'yellow',
            label = 'category_4')

plt.scatter(df_new[cluster_4_points, 0],
            df_new[cluster_4_points, 1],
            s =100,
            c = 'magenta',
            label = 'category_5')

plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()

## bifurcating the clusters further

kmeans = KMeans(n_clusters = 7, random_state = 42)
y_pred = kmeans.fit_predict(df_new)

df['result_new'] = y_pred
df.head(15)
df['result_new'].value_counts()
df['result_new'].unique()


df.groupby('result_new')['Spending Score (1-100)'].mean().plot(kind = 'bar')


df.groupby('result_new')['Annual Income (k$)'].mean().plot(kind = 'bar')



cluster_0_points = df[df['result_new']==0].index
cluster_1_points = df[df['result_new']==1].index
cluster_2_points = df[df['result_new']==2].index
cluster_3_points = df[df['result_new']==3].index
cluster_4_points = df[df['result_new']==4].index
cluster_5_points = df[df['result_new']==5].index
cluster_6_points = df[df['result_new']==6].index

kmeans.cluster_centers_    ## centroid values

## Visualising the clusters
plt.scatter(df_new[cluster_0_points, 0],
            df_new[cluster_0_points, 1],
            s =100,
            c = 'cyan',
            label = 'category_1')

plt.scatter(df_new[cluster_1_points, 0],
            df_new[cluster_1_points, 1],
            s =100,
            c = 'red',
            label = 'category_2')


plt.scatter(df_new[cluster_2_points, 0],
            df_new[cluster_2_points, 1],
            s =100,
            c = 'blue',
            label = 'category_3')

plt.scatter(df_new[cluster_3_points, 0],
            df_new[cluster_3_points, 1],
            s =100,
            c = 'yellow',
            label = 'category_4')

plt.scatter(df_new[cluster_4_points, 0],
            df_new[cluster_4_points, 1],
            s =100,
            c = 'magenta',
            label = 'category_5')

plt.scatter(df_new[cluster_5_points, 0],
            df_new[cluster_5_points, 1],
            s =100,
            c = 'black',
            label = 'category_6')


plt.scatter(df_new[cluster_6_points, 0],
            df_new[cluster_6_points, 1],
            s =100,
            c = 'pink',
            label = 'category_7')

kmeans.cluster_centers_

plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s =100,
            c = 'brown',
            label = 'centroid')


plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()

y_pred = kmeans.predict(df_new[-10:])
y_pred

kmeans.predict([[137,83]])


