import sys
import time
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, SpectralClustering


"""
This function implements the elbow method to find the optimal number of clusters.
"""
def elbow_method(model_name, data):
    # perform elbow method
    print("Performing the elbow method ... ")
    start = time.time()
    sse = []
    for k in range(1, 11):
        if model_name == 'kmeans':
            model = KMeans(n_clusters=k, random_state=0, n_init="auto")
        elif model_name == 'spectral':
            model = SpectralClustering(n_clusters=k, assign_labels='cluster_qr', random_state=0)
        else:
            print("Wrong model name!")
            break
        model.fit(data)
        sse.append(model.inertia_)
    print("Elbow method finished after", time.time() - start)

    # visualize results
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), sse)
    plt.xticks(range(1, 11))
    plt.title("Elbow Plot")
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    filename = '../images/elbow_' + model_name + '.png'
    plt.savefig(filename)


"""
This function implements the silhouette score method to find the optimal number of clusters.
"""
def silhouette_method(model_name, data):
    # perform silhouette method
    print("Performing the silhouette method ... ")
    start = time.time()
    sil = []
    for k in range(2, 11): # 1 not allowed
        if model_name == 'kmeans':
            model = KMeans(n_clusters=k, random_state=0, n_init="auto")
        elif model_name == 'spectral':
            model = SpectralClustering(n_clusters=k, assign_labels='cluster_qr', random_state=0)
        else:
            print("Wrong model name!")
            break
        model.fit(data)
        sil.append(silhouette_score(data, model.labels_, metric='euclidean'))
    print("Silhouette method finished after", time.time() - start)

    # visualize results
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, 11), sil)
    plt.xticks(range(2, 11))
    plt.title("Silhouette Plot")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette score")
    filename = '../images/silhouette_' + model_name + '.png'
    plt.savefig(filename)


"""
This function perform the clustering based on the corresponding initialized model
"""
def perform_clustering(model, data, user_id):
    y = model.fit_predict(data)
    model_data = data.copy()
    model_data['cluster'] = y
    model_data = pd.concat([user_id, model_data], axis=1, ignore_index=True)

    return model_data


if __name__ == '__main__':
    # read arguments from command line
    input_file = sys.argv[1]

    # load the input dataframe
    data = pd.read_pickle(input_file)

    # prepare data for clustering (store and then remove id)
    user_id = data['id']
    data.drop(columns=['id'], inplace=True)

    # find optimal k for spectral with the elbow method
    elbow_method('spectral', data)

    # find optimal k for spectral with the silhouette method
    silhouette_method('spectral', data)

    # perform spectral clustering
    start = time.time()
    print("Clustering with Spectral ... ")
    spectral = SpectralClustering(n_clusters=2, assign_labels='cluster_qr', random_state=0)
    results = perform_clustering(spectral, data, user_id)
    print("Spectral finished after", time.time() - start)
    results.to_csv('../data/clustering_results/spectral_results.csv')
