import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

class PriceMovement:
    def __init__(self, data):
        self.data = data

    def categorize(self):
        self.data['PriceChange'] = self.data['close'].diff()
        conditions = [
        (self.data['PriceChange'] > 0.5),
        (self.data['PriceChange'] < -0.5),
        (self.data['PriceChange'] >= -0.5) & (self.data['PriceChange'] <= 0.5)
    ]
        choices = ['Up', 'Down', 'Stable']
        self.data['PriceMovement'] = np.select(conditions, choices)
        self.data['PriceMovement'] = pd.Categorical(self.data['PriceMovement'], categories=choices, ordered=True)

class Volume:
    def __init__(self, data):
        self.data = data

    def categorize(self):
        conditions = [
            (self.data['volume'] <= self.data['volume'].quantile(0.33)),
            (self.data['volume'] <= self.data['volume'].quantile(0.67)),
            (self.data['volume'] > self.data['volume'].quantile(0.67))
        ]
        choices = ['Low', 'Medium', 'High']
        self.data['Volume'] = pd.cut(self.data['volume'], bins=[0] + list(self.data['volume'].quantile([0.33, 0.67, 1.0])), labels=choices, include_lowest=True)

def plot_all_kmeans(features_scaled, n_features):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('K-Means Clustering with Different k Values', fontsize=16)
    ax = axes.ravel()

    for i, k in enumerate(range(93, 100)):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(features_scaled)
        centroids = kmeans.cluster_centers_

        # Assume the first two features are 'close' and 'volume' after scaling and encoding
        ax[i].scatter(features_scaled[:, 0], features_scaled[:, 1], c=labels, cmap='viridis', alpha=0.5)
        ax[i].scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=100)
        ax[i].set_title(f'k={k}')
        ax[i].set_xlabel('Feature 1')
        ax[i].set_ylabel('Feature 2 (Scaled Volume)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def main():
    df = pd.read_csv('nflx_2014_2023.csv')
    df['date'] = pd.to_datetime(df['date'])

    # Instantiate and apply categorization
    pm = PriceMovement(df)
    pm.categorize()
    volm = Volume(df)
    volm.categorize()

    # Print the number of members in each class
    print("Class distribution in PriceMovement:")
    print(df['PriceMovement'].value_counts())

    print("Class distribution in Volume:")
    print(df['Volume'].value_counts())

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Date vs. Volume with color bar
    scatter = axes[0].scatter(df['date'], df['volume'], c=df['Volume'].cat.codes, cmap='viridis', alpha=0.6)
    axes[0].set_title('Date vs. Volume')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Volume')
    axes[0].set_yscale('log')
    axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)
    colorbar = fig.colorbar(scatter, ax=axes[0])
    colorbar.set_label('Volume Category')
    colorbar.set_ticks([0, 1, 2])
    colorbar.set_ticklabels(['Low', 'Medium', 'High'])

    # Plot 2: Date vs. Close Price by Price Movement
    colors = {'Up': 'blue', 'Down': 'orange', 'Stable': 'green'}
    for category, color in colors.items():
        subset = df[df['PriceMovement'] == category]
        axes[1].scatter(subset['date'], subset['close'], label=category, color=color, alpha=0.6)
    axes[1].set_title('Date vs. Close Price')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Close Price')
    axes[1].legend(title='Price Movement')
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)

    # Plot 3: Price Movement vs. Volume
    colors = ['blue', 'orange', 'green']  # Color mapping for price movements
    categories = df['PriceMovement'].cat.categories
    for category, color in zip(categories, colors):
        subset = df[df['PriceMovement'] == category]
        axes[2].scatter(subset['Volume'].cat.codes, subset['volume'], label=category, color=color, alpha=0.6)
    axes[2].set_title('Price Movement vs. Volume')
    axes[2].set_xlabel('Price Movement Category')
    axes[2].set_ylabel('Volume')
    axes[2].legend()

    plt.tight_layout()
    plt.show()

     # Creating a column transformer for both numerical and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['close', 'volume']),
            ('cat', OneHotEncoder(), ['PriceMovement', 'Volume'])
        ])

    # Fit and transform the data
    features_scaled = preprocessor.fit_transform(df)

    # The number of features after encoding
    n_features = features_scaled.shape[1]

    # K-Means experiments and plotting
    print("K-Means Clustering Results:")
    for k in range(93, 100):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(features_scaled)
        silhouette = silhouette_score(features_scaled, labels)
        print(f"Number of Clusters: {k}, Silhouette Score: {silhouette:.2f}")

    # Plot K-Means results for all k values considering the new number of features
    plot_all_kmeans(features_scaled, n_features)

    # Hierarchical clustering experiments
    print("\nHierarchical Clustering Results:")
    linkages = ['ward', 'complete', 'average']
    for linkage_type in linkages:
        cluster = AgglomerativeClustering(n_clusters=3, linkage=linkage_type)
        labels = cluster.fit_predict(features_scaled)
        print(f"Linkage Type: {linkage_type}, Cluster Labels: {np.unique(labels)}")

    # Optionally, visualize the dendrogram for one of the linkage methods
    plt.figure(figsize=(14, 7))  # Adjust the figure size to improve readability
    Z = linkage(features_scaled, method='ward')
    dendrogram(Z)
    plt.title('Dendrogram for Hierarchical Clustering')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.show()

if __name__ == "__main__":
    main()