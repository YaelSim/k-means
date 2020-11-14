import numpy as np
import scipy.io.wavfile
import sys
import math


def calculate_distance(p1, p2):
    # this func calculate the distance between 2 points
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def assign_points(centroids, points_array):
    # assign each point to a cluster by the closest centroid
    clusters_dict = {}
    # initialize the dictionary
    for i in range(len(centroids)):
        clusters_dict.setdefault(i, [])

    for x in points_array:
        dist_array = []
        for cen in centroids:
            # list of the distance from point x to all the centroids
            dist_array.append(calculate_distance(cen, x))
        # get the index of the minimum distance by centroid, so we know where to insert the point in the cluster
        key = np.argmin(dist_array)
        # ***clusters_dict = add_point_in_dict(clusters_dict, key, x)***********************
        clusters_dict[key].append(x)
    return clusters_dict


def update_centroids(dict_of_clusters):
    # this func calculate the average of each cluster and it will be the new centroid
    new_centroids = []
    for key in dict_of_clusters:
        cluster = dict_of_clusters.get(key)
        sum_array = np.sum(cluster, axis=0)
        point_x = sum_array[0] / float(len(cluster))
        point_y = sum_array[1] / float(len(cluster))
        sum_array[0], sum_array[1] = round(point_x), round(point_y)
        new_centroids.append(sum_array)
    return new_centroids


def main():
    sample, centroids = sys.argv[1], sys.argv[2]
    fs, y = scipy.io.wavfile.read(sample)  # reading
    x = np.array(y.copy())
    centroids = np.loadtxt(centroids)  # load the centroids to array

    iter_count = 0  # a counter for 30 iterations

    # open file
    out_file = open("output.txt", "w")

    # loop that iterate 30 times or convergence
    while iter_count < 30:
        dict_of_clusters = {}
        new_centroids = []

        # assign each point to the closest centroid
        dict_of_clusters = assign_points(centroids, x)

        # update each centroid to be the average of the point in this cluster
        new_centroids = update_centroids(dict_of_clusters)

        # write the centroids to the file
        out_file.write(f"[iter {iter_count}]:{', '.join([str(j) for j in new_centroids])}\n")

        # stop until got 30 or the update didn't changed
        if np.array_equal(centroids, new_centroids):
            break

        centroids = new_centroids
        iter_count += 1


if __name__ == "__main__":
    main()