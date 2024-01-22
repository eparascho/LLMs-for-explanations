import pandas as pd
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


if __name__ == '__main__':
    # read data
    data = pd.read_pickle('../data/clustering_df.pkl')

    # prepare data for clustering (store and then remove id)
    user_id = data['id']
    data.drop(columns=['id'], inplace=True)

    # visualize data through TSNE
    projection = TSNE().fit_transform(data)
    plt.scatter(*projection.T)
    plt.show()
