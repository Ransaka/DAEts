import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from yellowbrick.cluster import InterclusterDistance


def plot_kmeans_pca(data, n_clusters):
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters).fit(data)
    labels = kmeans.predict(data)

    # Perform PCA with n_components=2
    pca = PCA(n_components=2).fit(data)
    data_2d = pca.transform(data)

    # Create a scatter plot of the data colored by KMeans clusters
    plt.figure(figsize=(15,8))
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis')
    plt.show()
    
    plot_intercluster_distance(data, labels,n_clusters=n_clusters)

    return labels

def plot_time_series(df):
    # Get the unique labels
    labels = df['cluster'].unique()

    # Set the figure size
    plt.figure(figsize=(15, 8))

    colors = ['red', 'blue', 'green', 'orange', 'purple']

    # Loop through the labels
    for i, label in enumerate(labels):
        # Select the rows with the current label
        data = df[df['cluster'] == label]

        # Sample 5 users from the data
        users = data.sample(n=5).reset_index(drop=True)
        # Loop through the users

        plt.subplot(len(labels), 1, i+1)
        # Set the color
        alpha = np.linspace(0, 1, 5)
        color = colors[i % len(colors)]
        for j, user in users.iterrows():
            # Get the time series data for the user
            time_series = user[0:-1]
            # Plot the time series data
            plt.plot(time_series, alpha=alpha[j], color=color)
    
    # Show the plot
    plt.show()
    
    
def plot_intercluster_distance(X, labels, n_clusters):
    # Create the InterclusterDistance visualizer
    visualizer = InterclusterDistance(KMeans(n_clusters))

    # Fit and transform the data using the K-Means model
    visualizer.fit(X)

#     # Create a new figure
#     fig, ax = plt.subplots()

    # Plot the intercluster distance
    visualizer.show(outliers=True)