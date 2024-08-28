import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import statistics as stat
import seaborn as sns


#Read The Dataset#
dataset=pd.read_csv("C:/Users/uufci/Downloads/DataForTable2.1WHR2023.csv")

#Data Preprocessing#
dataset.info()
dataset.head()
#check for duplicated records#
duplicated_rows = dataset[dataset.duplicated()]
print("dupliacted rows are:" , duplicated_rows)

#Replace the missing values with the values from the previous year#
sorted_DS = dataset.sort_values(['Country name','year'])
sorted_DS['Log GDP per capita']= sorted_DS.groupby('Country name')['Log GDP per capita'].ffill()
sorted_DS['Social support']= sorted_DS.groupby('Country name')['Social support'].ffill()
sorted_DS['Healthy life expectancy at birth']= sorted_DS.groupby('Country name')['Healthy life expectancy at birth'].ffill()
sorted_DS['Freedom to make life choices']= sorted_DS.groupby('Country name')['Freedom to make life choices'].ffill()
sorted_DS['Generosity']= sorted_DS.groupby('Country name')['Generosity'].ffill()
sorted_DS['Perceptions of corruption']= sorted_DS.groupby('Country name')['Perceptions of corruption'].ffill()
sorted_DS['Positive affect']= sorted_DS.groupby('Country name')['Positive affect'].ffill()
sorted_DS['Negative affect']= sorted_DS.groupby('Country name')['Negative affect'].ffill()


#Replace the missing values with the value of the following year#
sorted_DS['Log GDP per capita']= sorted_DS.groupby('Country name')['Log GDP per capita'].bfill()
sorted_DS['Social support']= sorted_DS.groupby('Country name')['Social support'].bfill()
sorted_DS['Healthy life expectancy at birth']= sorted_DS.groupby('Country name')['Healthy life expectancy at birth'].bfill()
sorted_DS['Freedom to make life choices']= sorted_DS.groupby('Country name')['Freedom to make life choices'].bfill()
sorted_DS['Generosity']= sorted_DS.groupby('Country name')['Generosity'].bfill()
sorted_DS['Perceptions of corruption']= sorted_DS.groupby('Country name')['Perceptions of corruption'].bfill()
sorted_DS['Positive affect']= sorted_DS.groupby('Country name')['Positive affect'].bfill()
sorted_DS['Negative affect']= sorted_DS.groupby('Country name')['Negative affect'].bfill()
sorted_DS.info()

#Correlation Matrix#
selected_features = ['Life Ladder','Log GDP per capita','Social support','Healthy life expectancy at birth','Freedom to make life choices',
                     'Generosity','Perceptions of corruption','Positive affect','Negative affect']

selected_ds = dataset[selected_features]
correlation_matrix = selected_ds.corr()
sns.set(context='paper', font_scale=1)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix",color='black',size=15)
plt.show()

#Remove The below values as they are 2 records#
sorted_DS.dropna(subset='Social support',inplace=True)
sorted_DS.dropna(subset='Negative affect',inplace=True)
sorted_DS.dropna(subset='Positive affect',inplace=True)


#Apply Linear Regression to replace null values in GDP & Healthy life expectancy at birth missing values#
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#GDP#
X_train, X_test, y_train, y_test = train_test_split(sorted_DS.dropna(subset=['Log GDP per capita', 'Life Ladder']), 
                                                    sorted_DS['Log GDP per capita'].dropna(), test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train[['Life Ladder']], y_train)
sorted_DS.loc[sorted_DS['Log GDP per capita'].isnull(), 'Log GDP per capita'] = model.predict(sorted_DS[sorted_DS['Log GDP per capita'].isnull()][['Life Ladder']])

#Healthy Life Expectancy#
X_train, X_test, y_train, y_test = train_test_split(sorted_DS.dropna(subset=['Healthy life expectancy at birth', 'Life Ladder']), 
                                                  sorted_DS['Healthy life expectancy at birth'].dropna(), test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train[['Life Ladder']], y_train)
sorted_DS.loc[sorted_DS['Healthy life expectancy at birth'].isnull(), 'Healthy life expectancy at birth'] = model.predict(
    sorted_DS[sorted_DS['Healthy life expectancy at birth'].isnull()][['Life Ladder']])


#Apply Median to the rest of the missing value#
sorted_DS['Generosity'].fillna(sorted_DS['Generosity'].median(), inplace=True)
sorted_DS['Perceptions of corruption'].fillna(sorted_DS['Perceptions of corruption'].median(), inplace=True)
sorted_DS.info()

#Data Visualization#
x1= sorted_DS['Log GDP per capita']
x2= sorted_DS['Social support']
x3= sorted_DS['Healthy life expectancy at birth']

y = sorted_DS['Life Ladder']
plt.ylabel("Happiness Score")

plt.scatter(x1,y)
plt.xlabel("Log GDP per capita")
plt.show()

plt.scatter(x2,y)
plt.ylabel("Happiness Score")
plt.xlabel("Social support")
plt.show()

plt.scatter(x3,y)
plt.ylabel("Happiness Score")
plt.xlabel("Healthy life expectancy")
plt.show()

fig=plt.figure(figsize=(10,10),dpi=100,facecolor='lightgrey',clear=True)
Germany = sorted_DS[sorted_DS['Country name']=='Germany']
x= Germany['year']
y = Germany['Life Ladder']
plt.xlabel('Year', size=10)
plt.ylabel('Happiness Level',size=10)
plt.title("Happiness scores in Germany", size=15)
plt.scatter(x,y,color='darkblue')
plt.show()

Sorted_Level = dataset.sort_values('Life Ladder', ascending=False)
Year_2022 = Sorted_Level[Sorted_Level['year']==2022].iloc[:40]
fig=plt.figure(figsize=(10,10),dpi=100,facecolor='lightGrey',clear=True)
plt.barh(Year_2022['Country name'],Year_2022['Life Ladder'],color='lightblue')
plt.xlabel('Happiness Score',fontsize=16)
plt.ylabel('Country',fontsize=16)
plt.title('Highest Happiness scores 2022', fontsize=25)
plt.gca().invert_yaxis()
plt.show()

Sorted_Level = dataset.sort_values('Life Ladder', ascending=True)
Year_2022 = Sorted_Level[Sorted_Level['year']==2022].iloc[:40]
fig=plt.figure(figsize=(10,10),dpi=100,facecolor='lightGrey',clear=True)
plt.barh(Year_2022['Country name'],Year_2022['Life Ladder'],color='lightblue')
plt.xlabel('Happiness Score',fontsize=16)
plt.ylabel('Country',fontsize=16)
plt.title('Lowest Happiness scores 2022', fontsize=25)
plt.gca().invert_yaxis()
plt.show()




Germany_GDP = sorted_DS[sorted_DS['Country name']=='Germany']
n = len(Germany_GDP)
num_bins = int(np.sqrt(n))
plt.hist(Germany_GDP['Log GDP per capita'], bins=n)
plt.show()


#Multiple Linear Regression model to predict Happiness level#
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

x=sorted_DS.iloc[:,3:6].values
y=sorted_DS['Life Ladder']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared (R²): {r2}')

import statsmodels.api as sm
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
print(model.summary())

#Apply K-means to find groups of countries close to happiness levels#
from sklearn.cluster import KMeans
Year_2022= sorted_DS[sorted_DS['year']==2022].iloc[:]
Relevant_Features = Year_2022[['Life Ladder','Log GDP per capita','Social support','Healthy life expectancy at birth',
                               'Freedom to make life choices']]
                            
scaler = StandardScaler()
scaled_features = scaler.fit_transform(Relevant_Features)

KM = KMeans(n_clusters=2)
Relevant_Features['cluster']= KM.fit_predict(scaled_features)

sns.set(style="ticks")
sns.pairplot(Relevant_Features, hue='cluster', palette='viridis')
plt.suptitle("Pair Plot of Features with Cluster Coloring", y=1.02)
plt.show()

#plt.scatter(Year_2022['Life Ladder'], Year_2022['Social support'], c=Year_2022['cluster'], cmap='viridis', alpha=0.8)
#plt.title('K-means Clustering of Countries Based on Happiness Features')
#plt.xlabel('Life Ladder')
#plt.ylabel('Log GDP per capita')

print("Cluster Centers:")
print(KM.cluster_centers_)
print("\nCluster Assignments:")
#clusters = Year_2022[['Country name', 'cluster']].drop_duplicates()
#print(clusters)
#plt.show()

from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score

db_index = davies_bouldin_score(Relevant_Features, KM.labels_)
print(f"Davies-Bouldin Index: {db_index}")

silhouette_avg = silhouette_score(Relevant_Features, KM.labels_)
print(f"Silhouette Score: {silhouette_avg}")


