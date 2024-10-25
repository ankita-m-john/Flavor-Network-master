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