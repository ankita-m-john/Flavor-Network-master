 #!/usr/bin/env python3
 # -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS, TSNE
from matplotlib.colors import to_hex
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid

def tsne_cluster_cuisine(df,sublist):
    lenlist=[0]
    df_sub = df[df['cuisine']==sublist[0]]
    lenlist.append(df_sub.shape[0])
    for cuisine in sublist[1:]:
        temp = df[df['cuisine']==cuisine]
        df_sub = pd.concat([df_sub, temp],axis=0,ignore_index=True)
        lenlist.append(df_sub.shape[0])
    df_X = df_sub.drop(['cuisine','recipeName'],axis=1)

    dist = squareform(pdist(df_X, metric='cosine'))
    tsne = TSNE(metric='precomputed', init='random', perplexity=100, learning_rate=200, max_iter=1000).fit_transform(dist)

    palette = sns.color_palette("hls", len(sublist))
    palette_hex = [to_hex(color) for color in palette]
    
    plt.figure(figsize=(10,10))
    for i,cuisine in enumerate(sublist):
        plt.scatter(tsne[lenlist[i]:lenlist[i+1],0],
        tsne[lenlist[i]:lenlist[i+1],1],c=palette_hex[i],label=sublist[i])
    plt.legend()
    plt.title("t-SNE Clustering of Cuisines")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()

def elbow_Sil():
    # Elbow method
    inertia = []
    K_range = range(2, 21)  # Testing K from 2 to 20

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_ingr_scaled)
        inertia.append(kmeans.inertia_)

    # Plotting the elbow method
    plt.figure(figsize=(10, 5))
    plt.plot(K_range, inertia, marker='o')
    plt.title('Elbow Method for Ingredients:')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Inertia')
    plt.grid()
    plt.xticks(K_range)
    plt.show()

    # Silhouette analysis
    silhouette_scores = []

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(df_flavor_scaled)
        silhouette_avg = silhouette_score(df_flavor_scaled, cluster_labels)
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

    #For flavours:
    # Elbow method
    inertia = []
    K_range = range(2, 21)  # Testing K from 2 to 20

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_flavor_scaled)
        inertia.append(kmeans.inertia_)

    # Plotting the elbow method
    plt.figure(figsize=(10, 5))
    plt.plot(K_range, inertia, marker='o')
    plt.title('Elbow Method for Ingredients:')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Inertia')
    plt.grid()
    plt.xticks(K_range)
    plt.show()

    # Silhouette analysis
    silhouette_scores = []
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(df_ingr_scaled)
        silhouette_avg = silhouette_score(df_ingr_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    # Plotting silhouette scores
    plt.figure(figsize=(10, 5))
    plt.plot(K_range, silhouette_scores, marker='o')
    plt.title('Silhouette Analysis for Flavours')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.grid()
    plt.xticks(K_range)
    plt.show()

def pca(num):
    
    #For ingredients:
    X = df_ingr
    # Using PCA from sklearn PCA
    pca = PCA(n_components=0.95)
    X_centered = X - X.mean(axis=0)
    pca.fit(X_centered)
    X_pca = pca.transform(X_centered)
    if (num == 1):
        pca = PCA(n_components=2)
        X_centered = X - X.mean(axis=0)
        pca.fit(X_centered)
        X_pca = pca.transform(X_centered)
        return X_pca
    # Calculate the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    # Cumulative explained variance
    cumulative_variance = np.cumsum(explained_variance_ratio)
    # Determine the number of components needed to explain 95% of the variance
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1  # Add 1 because np.argmax returns 0-based index
    # Print the number of components needed to explain 95% variability
    print(f"\nNumber of components required to explain 95% variability: {n_components_95}")

    # Plot the explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_variance, marker='o', linestyle='--')
    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.title('Cumulative Explained Variance by PCA Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid()
    plt.show()
    
    #For flavors:
    X = df_flavor
    # Using PCA from sklearn PCA
    pca = PCA(n_components=0.95)
    X_centered = X - X.mean(axis=0)
    pca.fit(X_centered)
    X_pca = pca.transform(X_centered)
    if (num == 2):
        pca = PCA(n_components=2)
        X_centered = X - X.mean(axis=0)
        pca.fit(X_centered)
        X_pca = pca.transform(X_centered)
        return X_pca
    # Calculate the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    # Cumulative explained variance
    cumulative_variance = np.cumsum(explained_variance_ratio)
    # Determine the number of components needed to explain 95% of the variance
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1  # Add 1 because np.argmax returns 0-based index
    # Print the number of components needed to explain 95% variability
    print(f"\nNumber of components required to explain 95% variability: {n_components_95}")

    # Optional: Plot the explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_variance, marker='o', linestyle='--')
    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.title('Cumulative Explained Variance by PCA Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance of Flavour')
    plt.grid()
    plt.show()
    return X_pca

def GridS():
    # Define the parameter grid for eps and min_samples
    param_grid = {
        'eps': np.linspace(0.1, 1.0, 10),  # Epsilon values from 0.1 to 1.0
        'min_samples': range(1, 11)        # min_samples values from 1 to 10
    }

    # Initialize variables to store the best score and parameters
    best_score = -1
    best_params = None
    X_pca = pca(1)
    # Iterate over all combinations of eps and min_samples
    for params in ParameterGrid(param_grid):
        # Create a DBSCAN model with the current parameters
        model = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
        # Fit the model to the data
        dbscan_labels = model.fit_predict(X_pca)
        # Ignore noise points (labeled as -1)
        if len(set(dbscan_labels)) > 1:  # Ensure at least one cluster is formed
            score = silhouette_score(X_pca, dbscan_labels)
            print(f"eps: {params['eps']:.2f}, min_samples: {params['min_samples']}, silhouette score: {score:.4f}")

            # Update the best score and parameters if the current score is better
            if score > best_score:
                best_score = score
                best_params = params

    # Print the best parameters and score
    print(f"Best score: {best_score:.4f} for eps: {best_params['eps']:.2f} and min_samples: {best_params['min_samples']}")
    return(best_params['eps'], best_params['min_samples'])

def K_Means():
    # Perform K-Means clustering with k=7
    k = 8
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
    # Reduce the data dimensions using PCA for visualization
    X_pca = pca(1)
    kmeans.fit(X_pca)
    labels = kmeans.labels_
    
    # Plot the clustering results
    plt.figure(figsize=(10, 7))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='Set1', s=50, marker='o', alpha=0.7)
    plt.title(f'K-Means Clustering with k={k}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Cluster Label')
    plt.grid(True)
    plt.show()

def DBScan():
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_ingr)
    
    # Dimensionality reduction using t-sne for high density data
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    ideal_eps, ideal_min = GridS()
    # Fit DBSCAN
    dbscan = DBSCAN(eps=ideal_eps, min_samples=ideal_min)  # Adjust eps and min_samples as needed
    dbscan_labels = dbscan.fit_predict(X_tsne)

    # Handling noise points (label -1 represents noise in DBSCAN)
    unique_labels = np.unique(dbscan_labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    print(f"Number of clusters (excluding noise): {n_clusters}")
    print(f"Noise points: {sum(dbscan_labels == -1)}")

    # Visualize the DBSCAN results
    plt.figure(figsize=(10, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=dbscan_labels,  cmap='Set1', marker='o')
    plt.title("DBSCAN Clustering")
    plt.xlabel("Ingredients")
    plt.ylabel("Cuisine")
    plt.colorbar(label='Cluster Label')
    plt.show()

def GMM():
    n_components_range = range(1, 25)
    log_likelihoods = []
    X_pca = pca(1)
    for n in n_components_range:
        gmm = GaussianMixture(n_components=n, random_state=42)
        gmm.fit(X_pca)
        log_likelihoods.append(gmm.score(X_pca))  # Log-likelihood for this model

    plt.plot(n_components_range, log_likelihoods, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Log-Likelihood')
    plt.title('Elbow Method for Choosing n_components')
    plt.show()
    optimal_k = 8
    # GMM Clustering
    gmm = GaussianMixture(n_components=8, random_state=42, init_params='random', covariance_type='full')
    gmm_labels = gmm.fit_predict(X_pca)
    print(X_pca.dtype)
    # Visualize GMM results
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=gmm_labels, palette='Set1')
    plt.title('GMM Clustering Results')
    plt.show()


if __name__ == '__main__':
    yum_ingr = pd.read_pickle('/Users/ankita/Downloads/Flavor-Network-master/data/yummly_ingr.pkl')
    yum_ingrX = pd.read_pickle('/Users/ankita/Downloads/Flavor-Network-master/data/yummly_ingrX.pkl')
    yum_tfidf = pd.read_pickle('/Users/ankita/Downloads/Flavor-Network-master/data/yum_tfidf.pkl')
    
    #select all unique cuisines and plot tsne clustering with ingredients
    sublist = yum_ingr['cuisine'].unique()
    df_ingr = yum_ingrX.copy()
    df_ingr['cuisine'] = yum_ingr['cuisine']
    df_ingr['recipeName'] = yum_ingr['recipeName']
    # tsne_cluster_cuisine(df_ingr,sublist)

    #select all unique cuisines and plot tsne clustering with flavor
    sublist = yum_ingr['cuisine'].unique()
    df_flavor = yum_tfidf.copy()
    df_flavor['cuisine'] = yum_ingr['cuisine']
    df_flavor['recipeName'] = yum_ingr['recipeName']
    # tsne_cluster_cuisine(df_flavor,sublist)
    
    # Drop non-numeric columns if any, like 'cuisine' or 'recipeName'
    df_ingr = df_ingr.drop(columns=['cuisine', 'recipeName'], errors='ignore')  # Use errors='ignore' to skip if those columns don't exist
    df_ingr = pd.get_dummies(df_ingr, columns=df_ingr.columns, drop_first=True)
    df_flavor = df_flavor.drop(columns=['cuisine', 'recipeName'], errors='ignore')  # Use errors='ignore' to skip if those columns don't exist
    df_flavor = pd.get_dummies(df_flavor, columns=df_flavor.columns, drop_first=True)
    
    # Standardize the data (important for PCA)
    scaler = StandardScaler()
    df_ingr_scaled = scaler.fit_transform(df_ingr)
    scaler = StandardScaler()
    df_flavor_scaled = scaler.fit_transform(df_flavor)
    print("Got here")
    # X_pca = pca(0)
    # elbow_Sil()
    # K_Means()
    # DBScan()
    # GMM()