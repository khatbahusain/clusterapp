from sklearn.cluster import KMeans
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import silhouette_score


def kmeans(vectors, n_cluster):

    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(vectors)

    kmeans.labels_

    return kmeans.labels_



def silhouette(vectors, range_values):

    km_silhouette = []    
    range_values = [i for i in range(range_values[0], range_values[1])]


    for i in range_values:
        
        km = KMeans(n_clusters=i, random_state=0).fit(vectors)
        preds = km.predict(vectors)

        silhouette = silhouette_score(vectors, preds)

        km_silhouette.append(silhouette)
     

    
    # find the best number of clusters

    n_cluster = range_values[km_silhouette.index(max(km_silhouette))]

    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(vectors)
    clusters = kmeans.labels_

    return clusters, km_silhouette, n_cluster