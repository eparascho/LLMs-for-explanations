import time
from ClustersFeatures import *
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


"""
This function implements the elbow method to find the optimal number of clusters.
"""
def elbow_method(model_name, data_version, data):
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
    filename = '../images/elbow_' + model_name + '_' + data_version + '.png'
    plt.savefig(filename)
    plt.show()


"""
This function implements the silhouette score method to find the optimal number of clusters.
"""
def silhouette_method(model_name, data_version, data):
    # perform silhouette method
    print("Performing the silhouette method ... ")
    start = time.time()
    sil = []
    for k in range(2, 11):  # 1 not allowed
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
    filename = '../images/silhouette_' + model_name + '_' + data_version + '.png'
    plt.savefig(filename)
    plt.show()


"""
This function perform the clustering based on the corresponding initialized model
"""
def perform_clustering(model, data, metadata):
    y = model.fit_predict(data)
    model_data = data.copy()
    model_data['cluster'] = y
    model_data = pd.concat([metadata, model_data], axis=1)

    return model_data


"""
This function calculates and prints the clustering evaluation metrics
"""
def evaluation_metrics(data):
    # Silhouette score
    print("calculating Silhouette score ... ")
    print("The Silhouette score is:", silhouette_score(data.iloc[:, :-1], data['cluster']))

    # Davies-Bouldin Index
    print("calculating DB index ... ")
    print("The Davies-Bouldin Index is:", davies_bouldin_score(data.iloc[:, :-1], data['cluster']))

    # Calinski-Harabasz Index
    print("calculating CH index ... ")
    print("The Calinski-Harabasz Index is:", calinski_harabasz_score(data.iloc[:, :-1], data['cluster']))

    # Dunn Index
    print("calculating Dunn index ... ")
    features = data.iloc[:, :-1]
    centroids = features.groupby(data['cluster']).mean()
    intra_cluster_distances = features.groupby(data['cluster']).transform(lambda x: (x - x.mean()) ** 2)
    intra_cluster_distances = np.sqrt(intra_cluster_distances.sum(axis=1))
    inter_cluster_distances = pairwise_distances(centroids)
    np.fill_diagonal(inter_cluster_distances, np.inf)
    dunn_numerator = np.min(inter_cluster_distances)
    dunn_denominator = np.max(intra_cluster_distances.groupby(data['cluster']).max())
    dunn_index = dunn_numerator / dunn_denominator
    print("The Dunn Index is:", dunn_index)

    # PBM Index
    print("calculating PBM index ... ")
    overall_centroid = features.mean()
    total_intra_cluster_distance = intra_cluster_distances.sum()
    sum_squared_distances_to_centroid = ((features - overall_centroid) ** 2).sum(axis=1).sum()
    number_of_clusters = centroids.shape[0]
    E1 = sum_squared_distances_to_centroid
    Ek = total_intra_cluster_distance
    Dk = np.max(pairwise_distances(centroids, [overall_centroid]))
    pbm_index = (E1 * Dk / (Ek * number_of_clusters)) ** 2
    print("The PBM is:", pbm_index)

    # Xie-Beni Index
    print("calculating Xie-Beni index ... ")
    number_of_points = features.shape[0]
    min_inter_cluster_distance = np.min(inter_cluster_distances)
    xie_beni_index = total_intra_cluster_distance / (number_of_points * min_inter_cluster_distance)
    print("The Xie-Beni is:", xie_beni_index)
