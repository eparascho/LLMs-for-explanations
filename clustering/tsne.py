import pandas as pd
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


if __name__ == '__main__':
    # read data
    data = pd.read_csv('../data/clustering_results/kmeans_results_full.csv')

    # prepare data for clustering (store and then remove id)
    clusters = data['cluster']
    data.drop(columns=['id', 'date', 'cluster'], inplace=True)

    # coloring data points based on cluster
    # color_map = {0: '#9999ff', 1: '#ffff33', 2: '#99ffce', 3: '#ffb366'}  # for clean and categories
    color_map = {-1: '#9999ff', 0: '#ffff33', 1: '#99ffce', 2: '#ffb366', 3: '#66ccff'}  # for full
    colors = clusters.map(color_map)

    # visualize data through TSNE
    projection = TSNE().fit_transform(data)
    plt.scatter(*projection.T, s=10, c=colors)
    plt.show()
