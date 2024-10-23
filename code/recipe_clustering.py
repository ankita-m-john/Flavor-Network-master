'''After recipes have been mapped to ingredient space and flavor space
Plot tsne clustering
Plot with bokeh interactive plotting
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS, TSNE

from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool

#take some regional cuisines, tsne clustering, and plotting
def tsne_cluster_cuisine(df,sublist, perplexity_values, learning_rate_values):
    lenlist=[0]
    df_sub = df[df['cuisine']==sublist[0]]
    lenlist.append(df_sub.shape[0])
    for cuisine in sublist[1:]:
        temp = df[df['cuisine']==cuisine]
        df_sub = pd.concat([df_sub, temp],axis=0,ignore_index=True)
        lenlist.append(df_sub.shape[0])
    df_X = df_sub.drop(['cuisine','recipeName'],axis=1)
    print(df_X.shape, lenlist)

    dist = squareform(pdist(df_X, metric='cosine'))
    
    # Tune different values of perplexity and learning rate
    for perplexity in perplexity_values:
        for lr in learning_rate_values:
            tsne = TSNE(metric='precomputed', init='random', perplexity=perplexity, 
                        learning_rate=lr, max_iter=1000).fit_transform(dist)
            
    palette = sns.color_palette("hls", len(sublist))
    plt.figure(figsize=(10,10))
    for i,cuisine in enumerate(sublist):
        plt.scatter(tsne[lenlist[i]:lenlist[i+1],0],\
        tsne[lenlist[i]:lenlist[i+1],1],c=palette[i],label=sublist[i])
    plt.legend()
    plt.title("t-SNE Clustering of Cuisines")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()

#interactive plot with boken; set up for four categories, with color palette; pass in df for either ingredient or flavor
def plot_bokeh(df,sublist,filename):
    lenlist=[0]
    df_sub = df[df['cuisine']==sublist[0]]
    lenlist.append(df_sub.shape[0])
    for cuisine in sublist[1:]:
        temp = df[df['cuisine']==cuisine]
        df_sub = pd.concat([df_sub, temp],axis=0,ignore_index=True)
        lenlist.append(df_sub.shape[0])
    df_X = df_sub.drop(['cuisine','recipeName'],axis=1)
    print(df_X.shape, lenlist)

    dist = squareform(pdist(df_X, metric='cosine'))
    tsne = TSNE(metric='precomputed', init='random', perplexity=50, learning_rate=200, max_iter=1000).fit_transform(dist)
    #cannot use seaborn palette for bokeh
    palette =['red','green','blue','yellow']
    colors =[]
    for i in range(len(sublist)):
        for j in range(lenlist[i+1]-lenlist[i]):
            colors.append(palette[i])
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

    p = figure(width=1000, height=1000, tools=[hover],
               title="flavor clustering")

    p.circle('x', 'y', size=10, source=source, fill_color='colors')

    show(p)


if __name__ == '__main__':
    yum_ingr = pd.read_pickle('/Users/ankita/Downloads/Flavor-Network-master/data/yummly_ingr.pkl')
    yum_ingrX = pd.read_pickle('/Users/ankita/Downloads/Flavor-Network-master/data/yummly_ingrX.pkl')
    yum_tfidf = pd.read_pickle('/Users/ankita/Downloads/Flavor-Network-master/data/yum_tfidf.pkl')
    # Tune perplexity and learning rate values
    perplexity_values = [5, 30, 50]  # Example values
    learning_rate_values = [10, 200, 500]  # Example values
    #select four cuisines and plot tsne clustering with ingredients
    sublist = ['Italian','French','Japanese','Indian']
    df_ingr = yum_ingrX.copy()
    df_ingr['cuisine'] = yum_ingr['cuisine']
    df_ingr['recipeName'] = yum_ingr['recipeName']
    tsne_cluster_cuisine(df_ingr,sublist,perplexity_values, learning_rate_values)

    #select four cuisines and plot tsne clustering with flavor
    sublist = ['Indian', 'American']
    df_flavor = yum_tfidf.copy()
    df_flavor['cuisine'] = yum_ingr['cuisine']
    df_flavor['recipeName'] = yum_ingr['recipeName']
    tsne_cluster_cuisine(df_flavor,sublist,perplexity_values, learning_rate_values)

    #select four cuisines and do interactive plotting with bokeh
    # plot_bokeh(df_flavor,sublist, 'test1.html')
    # plot_bokeh(df_ingr,sublist, 'test2.html')
