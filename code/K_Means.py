import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the data
yum_ingr = pd.read_pickle('/Users/ankita/Downloads/Flavor-Network-master/data/yummly_ingr.pkl')
yum_ingrX = pd.read_pickle('/Users/ankita/Downloads/Flavor-Network-master/data/yummly_ingrX.pkl')

# Choose the dataset to cluster
df_X = yum_ingrX.copy()

# Add cuisine and recipeName for reference
df_X['cuisine'] = yum_ingr['cuisine']
df_X['recipeName'] = yum_ingr['recipeName']

# Remove the 'cuisine' and 'recipeName' columns before clustering
df_X_numeric = df_X.drop(columns=['cuisine', 'recipeName'])

# Check for missing values
df_X_numeric = df_X_numeric.fillna(0)  # Handle missing values

# Scale the numeric data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_X_numeric)  # Scale the features

# Perform K-Means clustering with k=7
k = 8
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)
labels = kmeans.labels_

# Add cluster labels to the original DataFrame
df_X['Cluster'] = labels

# Reduce the data to 2 dimensions using PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot the clustering results
plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50, marker='o', alpha=0.7)
plt.title(f'K-Means Clustering with k={k}')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster Label')
plt.grid(True)
plt.show()
