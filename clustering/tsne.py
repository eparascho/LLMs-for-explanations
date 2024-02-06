import pandas as pd
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


if __name__ == '__main__':
    # read data
    data = pd.read_pickle('../data/clustering_input/clustering_df_categories.pkl')

    # prepare data for clustering (store and then remove id)
    data.drop(columns=['id', 'date'], inplace=True)

    # visualize data through TSNE
    projection = TSNE().fit_transform(data)
    plt.scatter(*projection.T)
    plt.show()
