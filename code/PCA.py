import pandas as pd
import numpy as np
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

# Load your data (replace with your specific dataset)
yum_ingrX = pd.read_pickle('/Users/ankita/Downloads/Flavor-Network-master/data/yummly_ingrX.pkl')

# Step 1: Preprocess the Data
# Drop non-numeric columns if any, like 'cuisine' or 'recipeName'
df_X = yum_ingrX.drop(columns=['cuisine', 'recipeName'], errors='ignore')  # Use errors='ignore' to skip if those columns don't exist

# Standardize the data (important for PCA)
scaler = StandardScaler()
df_X_scaled = scaler.fit_transform(df_X)

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
