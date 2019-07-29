# ********************************************************************************************************
#                                    HELPER FUNCTIONS DEFINITION
# ********************************************************************************************************

import numpy as np
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from sklearn import datasets

# ********************************************
#             Distance Function
# ********************************************
def dist(pointA, pointB, order=None):
    return np.linalg.norm(pointA-pointB, ord=order)

# ********************************************
#          Dataset Creation Function
# ********************************************
def create_dataset(shape='gmm', sampleSize=200, n_clusters=3):
    clusterStd = [0.5, 1, 2]*3
    clusterStd = clusterStd[:n_clusters]
    
    if shape=='gmm':
        X = datasets.make_blobs(n_samples=sampleSize, n_features=2, centers=n_clusters, cluster_std=clusterStd)[0]
        X[:,0] = 30*(X[:,0]-min(X[:,0]))/(max(X[:,0])-min(X[:,0])) - 15
        X[:,1] = 30*(X[:,1]-min(X[:,1]))/(max(X[:,1])-min(X[:,1])) - 15
        
    elif shape=='circle':
        X = datasets.make_circles(n_samples=sampleSize, factor=.5, noise=.05)[0]
        X[:,0] = 30*(X[:,0]-min(X[:,0]))/(max(X[:,0])-min(X[:,0])) - 15
        X[:,1] = 30*(X[:,1]-min(X[:,1]))/(max(X[:,1])-min(X[:,1])) - 15
        
    elif shape=='moon':
        X = datasets.make_moons(n_samples=sampleSize, noise=.1)[0]
        X[:,0] = 30*(X[:,0]-min(X[:,0]))/(max(X[:,0])-min(X[:,0])) - 15
        X[:,1] = 30*(X[:,1]-min(X[:,1]))/(max(X[:,1])-min(X[:,1])) - 15

    elif shape=='anisotropic':
        transformations = {0:[[0.6, -0.6], [-0.4, 0.8]], 1:[[-0.7, -0.6], [0.6, 0.8]], 2:[[0.8, -0.1], [0.8, 0.1]]}
        X, y = datasets.make_blobs(n_samples=sampleSize, n_features=2, centers=n_clusters, cluster_std=clusterStd)
        for i in range(n_clusters):
            X[y==i] = np.dot(X[y==i], transformations[i%3])
        X = 5*X
        X[:,0] = 30*(X[:,0]-min(X[:,0]))/(max(X[:,0])-min(X[:,0])) - 15
        X[:,1] = 30*(X[:,1]-min(X[:,1]))/(max(X[:,1])-min(X[:,1])) - 15
    else:
        X = 30*np.random.rand(sampleSize, 2)-15
        
    return X


# ********************************************
#       Centroid Initialization Function
# ********************************************
def init_centroids(X, k=3, initMethod='random'):
    if initMethod == 'random':
        # Random Initialization
        centroids = 20*np.random.rand(k, 2)-10
    else:
        # Kmeans++ Initialization
        indices = list(range(X.shape[0]))
        centroids = np.empty((k,2))
        centroids[0, :] = X[np.random.choice(indices), :]
        D = np.power(np.linalg.norm(X-centroids[0], axis=1), 2)
        P = D/D.sum()

        for i in range(k-1):
            centroidIdx = np.random.choice(indices, size=1, p=P)
            centroids[i+1, :] = X[centroidIdx, :]
            Dtemp = np.power(np.linalg.norm(X-centroids[-1, :], axis=1), 2)
            D = np.min(np.vstack((Dtemp, D)), axis=0)
            P = D/D.sum()
        
    return centroids

# ********************************************
#      K-means Expectation Step Function
# ********************************************
def Kmeans_EStep(X, centroids):
    k = centroids.shape[0]
    # Initialize Points-Centroid Distance Matrix
    centroidDistMat = np.empty((X.shape[0], k))

    # Compute Points-Centroid Distance Matrix
    for i in range(k):
        centroidDistMat[:, i] = np.linalg.norm(X-centroids[i,:], axis=1)
    
    # Infer Labels
    labels = centroidDistMat.argmin(axis=1) 
    return labels


# ********************************************
#      K-means Maximization Step Function
# ********************************************
def Kmeans_MStep(X, centroids, labels, k):
    Newcentroids = centroids
    # Compute Values for New Centroids
    for i in range(k):
        if sum(labels==i)>0:
            Newcentroids[i, :] = X[labels==i, :].mean(axis=0)

    return Newcentroids


# ********************************************
#   Make K-means Visualization Data Function
# ********************************************
def make_kmeans_viz_data(X, labels, centroids, clusColorMap):

    clusLabelColors = list(map(clusColorMap.get, labels))
    
    # Num Clusters
    k = centroids.shape[0]

    data = []
    
    if sum(labels)==-X.shape[0]:
        # Data Points
        tracePoints = go.Scatter(
                    x = X[:, 0],
                    y = X[:, 1],
                    mode = 'markers',
                    marker = dict(color='gray', size=10,
                                  line = dict(width=1, color='white')),
                    name = 'Data Points'
        )
        data.append(tracePoints)
        # Centroid-Point Lines
        centroidPoints = X.copy()
        for idx,i in enumerate(range(1, 3*centroidPoints.shape[0], 3)):
            centroidPoints = np.insert(centroidPoints, i, X.mean(axis=0), axis=0)
            centroidPoints = np.insert(centroidPoints, i+1, np.array([None, None]), axis=0)
        traceLines = go.Scatter(
                        x = centroidPoints[:, 0],
                        y = centroidPoints[:, 1],
                        mode = 'lines',
                        line = dict(color='white', width=.5),
                        name = 'Memberships'
        )
        data.append(traceLines)
        # Centroids
        traceCentroid = go.Scatter(
                        x = centroids[:, 0],
                        y = centroids[:, 1],
                        name = 'Centroids',
                        mode='markers',
                        marker = dict(color='gray', 
                                      size=20, symbol='circle', opacity=.8, line = dict(width=3, color='black'))
        )
        data.append(traceCentroid)
        return data
    
    ## FIRST TRACE TYPE : DATA POINTS
    tracePoints = go.Scatter(
                    x = X[:, 0],
                    y = X[:, 1],
                    mode = 'markers',
                    marker = dict(color=[clusColorMap[i] for i in labels], size=10,
                                  line = dict(width=1, color='white')),
                    name = 'Data Points'
    )
    data.append(tracePoints)

    ## SECOND TRACE TYPE : LINES BETWEEN DATA POINTS AND CENTROIDS
        # Compute Array with Nones to link centroids to their respective points with lines
    centroidPoints = X.copy()
    for idx,i in enumerate(range(1, 3*centroidPoints.shape[0], 3)):
        centroidPoints = np.insert(centroidPoints, i, centroids[labels[idx], :], axis=0)
        centroidPoints = np.insert(centroidPoints, i+1, np.array([None, None]), axis=0)

        # Trace Lines
    traceLines = go.Scatter(
                    x = centroidPoints[:, 0],
                    y = centroidPoints[:, 1],
                    mode = 'lines',
                    line = dict(color='white', width=.5),
                    name = 'Cluster Lines'
    )
    data.append(traceLines)
    
    ## THIRD TRACE TYPE : CENTROIDS
    traceCentroid = go.Scatter(
                        x = centroids[:, 0],
                        y = centroids[:, 1],
                        name = 'Centroids',
                        mode='markers',
                        marker = dict(color=list(clusColorMap.values()), 
                                      size=20, symbol='circle', opacity=.8, line = dict(width=3, color='black'))
    )
    data.append(traceCentroid)

    return data
    