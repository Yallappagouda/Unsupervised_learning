import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os



df=pd.read_csv('Mall_Customers.csv')
df.head()
df.describe()

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]


scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

Kmeans= KMeans(n_clusters=5, random_state=42)
df['Cluster'] = Kmeans.fit_predict(X_scaled)

plt.figure(figsize=(8, 6)) 
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=df, palette='Set2')
plt.title('Customer Segmentation')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.grid()
plt.show()