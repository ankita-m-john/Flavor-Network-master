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
# from bokeh.palettes import Category10, Category20, Viridis256
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool, ColorBar
from matplotlib.colors import to_hex

def tsne_cluster_cuisine(df,sublist):
    lenlist=[0]
    df_sub = df[df['cuisine']==sublist[0]]
    lenlist.append(df_sub.shape[0])
    for cuisine in sublist[1:]:
        temp = df[df['cuisine']==cuisine]
        df_sub = pd.concat([df_sub, temp],axis=0,ignore_index=True)
        lenlist.append(df_sub.shape[0])
    df_X = df_sub.drop(['cuisine','recipeName'],axis=1)
    # print(df_X.shape, lenlist)

    dist = squareform(pdist(df_X, metric='cosine'))
    tsne = TSNE(metric='precomputed', init='random', perplexity=100, learning_rate=200, max_iter=1000).fit_transform(dist)

    palette = sns.color_palette("hls", len(sublist))
    palette_hex = [to_hex(color) for color in palette]
    
    plt.figure(figsize=(10,10))
    # plt.xaxis.major_label_orientation = 90
    for i,cuisine in enumerate(sublist):
        plt.scatter(tsne[lenlist[i]:lenlist[i+1],0],
        tsne[lenlist[i]:lenlist[i+1],1],c=palette_hex[i],label=sublist[i])
    plt.legend()
    plt.title("t-SNE Clustering of Cuisines")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()

#interactive plot with boken; set up for four categories, with color palette; pass in df for either ingredient or flavor
def plot_bokeh(df,sublist,filename):
    lenlist=[0]
    # print(len(sublist))
    df_sub = df[df['cuisine']==sublist[0]]
    lenlist.append(df_sub.shape[0])
    for cuisine in sublist[1:]:
        temp = df[df['cuisine']==cuisine]
        df_sub = pd.concat([df_sub, temp],axis=0,ignore_index=True)
        lenlist.append(df_sub.shape[0])
    df_X = df_sub.drop(['cuisine','recipeName'],axis=1)
    # print(df_X.shape, lenlist)

    dist = squareform(pdist(df_X, metric='cosine'))
    tsne = TSNE(metric='precomputed', init='random', perplexity=100, learning_rate=200, max_iter=1000).fit_transform(dist)
    #cannot use seaborn palette for bokeh
    palette = sns.color_palette("hls", len(sublist))
    palette_hex = [to_hex(color) for color in palette]
    # palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
    #            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # print(palette)

    colors =[]
    for i in range(len(sublist)):
        num_points_in_cuisine = lenlist[i+1] - lenlist[i]  # Get the number of points for this cuisine
        colors.extend([palette_hex[i]] * num_points_in_cuisine)
    # print(colors)
        # for _ in range(lenlist[i+1]-lenlist[i]):
        #     colors.append(palette[i])
        # unique_count = len(set(colors))
        # print(unique_count)  
    
    #plot with boken
    output_file(filename)
    source = ColumnDataSource(
            data=dict(x=tsne[:,0],y=tsne[:,1],
                cuisine = df_sub['cuisine'],
                colors = colors,
                recipe = df_sub['recipeName']))

    hover = HoverTool(tooltips=[
                ("cuisine", "@cuisine"),
                ("recipe", "@recipe")])

    p = figure(width=1000, height=1000, tools=[hover], title="flavor clustering")
    p.xaxis.major_label_orientation = 90
    # color_bar = ColorBar(color_mapper=color_mapper['transform'], width=8, location=(0,0))
    # p.add_layout(color_bar, 'right')

    p.circle('x', 'y', size=10, source=source, fill_color='colors')

    show(p)

def elbow(yum_ingr):
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

def pca(df_X_scaled):

    # Step 2: Apply PCA
    pca = PCA()
    pca.fit(df_X_scaled)
    
    # Step 3: Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    
    # Step 4: Determine how many components are needed for 95% variance
    cumulative_variance = np.cumsum(explained_variance)
    
    # Finding number of components to reach 95%
    num_components = np.argmax(cumulative_variance >= 0.95) + 1  # Adding 1 to get the count
    
    # Print the percentage of variance explained by each component
    for i, variance in enumerate(explained_variance):
        print(f"Component {i+1}: {variance:.4f} (or {variance*100:.2f}%)")
    
    # Print the number of components needed to explain 95% variance
    print(f"\nNumber of components needed to explain 95% of the variance: {num_components}")
    
    # Plotting the explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.axvline(x=num_components, color='g', linestyle='--')
    plt.title('Cumulative Explained Variance by PCA Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.xticks(np.arange(1, len(explained_variance) + 1, step=1))
    plt.grid()
    plt.show()

def K_Means(df_X_scaled):
    # Perform K-Means clustering with k=7
    k = 8
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
    kmeans.fit(df_X_scaled)
    labels = kmeans.labels_

    # Add cluster labels to the original DataFrame
    df_X['Cluster'] = labels

    # Reduce the data to 2 dimensions using PCA for visualization
    pca = PCA(n_components=208)
    X_pca = pca.fit_transform(df_X_scaled)

    # Plot the clustering results
    plt.figure(figsize=(10, 7))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50, marker='o', alpha=0.7)
    plt.title(f'K-Means Clustering with k={k}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Cluster Label')
    plt.grid(True)
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

    #select all unique cuisines and do interactive plotting with bokeh
    # plot_bokeh(df_flavor,sublist, 'test1.html')
    # plot_bokeh(df_ingr,sublist, 'test2.html')
    
    # Step 1: Preprocess the Data
    # Drop non-numeric columns if any, like 'cuisine' or 'recipeName'
    df_X = yum_ingrX.drop(columns=['cuisine', 'recipeName'], errors='ignore')  # Use errors='ignore' to skip if those columns don't exist
    df_X = pd.get_dummies(df_X, columns=df_X.columns, drop_first=True)
    # Standardize the data (important for PCA)
    scaler = StandardScaler()
    df_X_scaled = scaler.fit_transform(df_X)
    pca(df_X_scaled)
    elbow(yum_ingr)
    K_Means(df_X_scaled)
    