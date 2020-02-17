''''
Name: Srushti Kokare
K-means clustering for k from 1 to 10

'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

dataset = np.loadtxt("GMM_dataset.txt")

style.use('ggplot')
colorarray = ['r','b','y','m','g','c','k','m','w','brown']
list_of_sse = []
Optimal_SSE = 99999,


#this method gives the indices of the po
def find_nearest_centre(centers, dataset, k):
    idx_cluster = [[] for i in range(k)]
    for i in range(0, dataset.shape[0]):
        classified_indices = np.argmin(np.sqrt(np.sum(np.power(centers - dataset[i, :], 2), axis=1)))
        idx_cluster[classified_indices].append(i)
    return idx_cluster


#caculate sum sqaured error
def calculate_sse(cluster,centres):

    sum = 0
    for i in range(cluster.shape[0]):
        cluster[i,:] - centres
        sum = sum + (np.power((centres[0] - cluster[i, 0]),2) + np.power((centres[1] - cluster[i, 1] ),2))
    return sum

#find k centres till the model converges
def find_k_means(k):
    idx = np.random.randint(1500, size=k)
    centroids = dataset[idx,:]
    while True:
        cluster_indices = find_nearest_centre(centroids, dataset, k)
        k_new_centres = []
        for i in range(k):
            if(len(cluster_indices) != 0):
                new_centre = np.mean(dataset[cluster_indices[i], :], axis=0)
                k_new_centres.append(new_centre)
            else:
                k_new_centres.append(centroids[i,:])
        if np.sum(abs(centroids - k_new_centres)) == 0:
            break
        centroids = np.asarray(k_new_centres)

    for i in range(k):
        error = []
        error.append(calculate_sse(dataset[cluster_indices[i], :], k_new_centres[i]))

    SSE_error_k = np.sum(error)

    return cluster_indices, k_new_centres, SSE_error_k


#get the best clustering model from 10 iterations
def get_best_clustering(k):
    Optimal_SSE = float('Infinity')
    for i in range(10):			# Cluster the entire data 10 times
        cluster_indices, centres, sse = find_k_means(k)
        if sse < Optimal_SSE:
            final_cluster_indices = cluster_indices  	# Replace best_model with new best_model
            final_centres = centres
            final_sse = sse
            Optimal_SSE = sse
    #Print centroids
    for i in range(k):
        print("For k ",k,"centroid is ",final_centres[i-1])

    #Pring sum sqaured error
    print("SSE",final_sse)
    list_of_sse.append(final_sse)

    #Plot clusters with datapoints and centroid
    for i in range(k):
        cluster = dataset[final_cluster_indices[i],:]
        for j in range(cluster.shape[0]):
            plt.scatter(cluster[j,0], cluster[j,1],color = colorarray[i])
        plt.scatter(final_centres[i-1][0],final_centres[i-1][1] , s=130, marker="x", color='k')
    plt.show()


def main():
    for k in range(1,11):
        get_best_clustering(k)

    #Plot sse vs k
    plt.plot([1,2,3,4,5,6,7,8,9,10], list_of_sse)
    plt.show()
main()