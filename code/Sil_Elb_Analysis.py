import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Load the data
yum_ingr = pd.read_pickle('/Users/ankita/Downloads/Flavor-Network-master/data/yummly_ingr.pkl')
yum_ingrX = pd.read_pickle('/Users/ankita/Downloads/Flavor-Network-master/data/yummly_ingrX.pkl')
yum_tfidf = pd.read_pickle('/Users/ankita/Downloads/Flavor-Network-master/data/yum_tfidf.pkl')

# Choose the dataset to cluster
# For this example, let's use yum_ingrX as the feature set
sublist = yum_ingr['cuisine'].unique()
df_X = yum_ingrX.copy()

# df_X['cuisine'] = yum_ingr['cuisine']
# df_X['recipeName'] = yum_ingr['recipeName']
# Select numeric data only

# Check if the DataFrame is empty
if df_X.empty:
    print("DataFrame df_X is empty!")
else:
    print("Shape of df_X:", df_X.shape)
    print("First few rows of df_X:\n", df_X.head())

# Check for missing values
if df_X.isnull().sum().any():
    print("Missing values detected, filling NaN values with 0.")
    df_X = df_X.fillna(0)  # Handle missing values


# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_X)  # Scale the features

# Elbow method
inertia = []
K_range = range(2, 21)  # Testing K from 2 to 20

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plotting the elbow method
plt.figure(figsize=(10, 5))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.grid()
plt.xticks(K_range)
plt.show()

# Silhouette analysis
silhouette_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plotting silhouette scores
plt.figure(figsize=(10, 5))
plt.plot(K_range, silhouette_scores, marker='o')
plt.title('Silhouette Analysis for Optimal K')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Silhouette Score')
plt.grid()
plt.xticks(K_range)
plt.show()
