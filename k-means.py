"""
ISYE 7640 Homework 1
@Author: Siham Elmali

Step by step process for the k-means algorithm:
1- load the image to a numpy array
2- pick initial cluster centers from the data
3- calculate the distance from each pixel to the cluster centers
4- assign each pixel to a cluster based on the minimum distance (return the index)
5- calculate the mean of all of the pixels in each cluster
6- use the mean as the new cluster centers
7- repeat  steps 3 through 6 while checking for convergence ||old centers -  new centers||^2 < to a certain threshold)
"""

import cv2
import numpy as np
import time
import copy
import matplotlib.pyplot as plt
from numpy import array
import warnings


# ignore warnings
warnings.filterwarnings('ignore')


def read_image(file):
    """
    :param file: path to the image
    :return: 2-D matrix with 3 columns corresponding to the RGB values for each pixel
    """
    # Read the image
    img = cv2.imread(file)

    # convert bgr to rgb
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # turn the image to a numpy array of integer values
    im_arr = np.array(img_rgb, dtype='int32')

    # transform the img array from a "3-D" matrix to a flattened "2-D" matrix
    img_reshaped = im_arr.reshape((im_arr.shape[0] * im_arr.shape[1]), im_arr.shape[2])
    return img_reshaped


# 2- pick initial cluster centers from the data
def initialize_cluster_centers(img_matrix, num_clusters):
    """
    :param img_matrix: The 2D matrix with 3 columns for RGB values for each pixel
    :param num_clusters: The number of clusters (i.e K)
    :return: Picks random centers from the the pixel data
    """

    # Generate a uniform random sample from the pixel matrix of size num_clusters
    cluster_centers = img_matrix[np.random.choice(img_matrix.shape[0], size=num_clusters, replace=False), :]
    return cluster_centers


def pixels_to_cluster_center_distance(pixels, centers):
    """
    :param pixels: The 2D matrix with 3 columns for RGB values for each pixel
    :param centers: cluster centers
    :return: distance from each pixel to he cluster centers
    """
    num_pixels = pixels.shape[0]
    num_cluster_centers = centers.shape[0]

    distances = np.empty(shape=(num_pixels, num_cluster_centers))

    for i in range(num_pixels):
        for j in range(num_cluster_centers):
            # get the difference
            diff = pixels[i, :] - centers[j]

            # The dot product of the difference and itself
            distances[i, j] = np.dot(diff, diff)

    return distances


def cluster_assignment(distances):
    """
    :param distances: the distance of each pixel to the cluster centers
    :return: cluster assignment for each pixel
    """
    num_pixels, num_clusters = distances.shape
    cluster_labels = np.empty((num_pixels, 1))

    for i in range(num_pixels):
        # get the index of the minimum entree in each row in distances
        cluster_labels[i, :] = np.argmin(distances[i])

    return cluster_labels


def update_cluster_centers(pixels, labels, centers):
    """
    :param pixels:The 2D matrix with 3 columns for RGB values for each pixel
    :param labels: cluster assignment
    :param centers: cluster centers
    """
    k = centers.shape[0]
    num_pixels = pixels.shape[0]

    for j in range(k):
        mask = (labels[:, 0] == j)
        centers[j] = np.mean(pixels[mask], axis=0)


def k_means(pixels, k):
    """
    :param pixels: The 2D matrix with 3 columns for RGB values for each pixel
    :param k: number of clusters
    :return: cluster assignment and cluster centers
    """
    outer_loop_timer = time.time()
    centers = initialize_cluster_centers(pixels, k)

    max_iter = 200
    iter_count = 1
    has_converged = False
    convergence_threshold = 0.01
    labels = np.zeros(len(pixels))
    while (not has_converged) and (iter_count < max_iter):
        # start_loop = time.time()
        distance = pixels_to_cluster_center_distance(pixels, centers)
        labels = cluster_assignment(distance)

        # update centers
        old_centers = copy.deepcopy(centers)
        np.array(update_cluster_centers(pixels, labels, centers))

        # check convergence
        diff = array(centers) - array(old_centers)
        delta = np.linalg.norm(diff, ord=2)
        if delta < convergence_threshold:
            print('Center converged')
            has_converged = True

        iter_count += 1
    print("For k = {}, the total numbers of iterations till convergence is {}, and the total runtime is {} seconds".
          format(k, iter_count, round(time.time() - outer_loop_timer)))
    print("-----------------------------------------------------------------")

    return labels, centers


def compress_image(img, num_colors):
    """
    :param img: Image to be compressed
    :param num_colors: number of colors for compression
    :return: a compressed image reshape to the original dimensions
    """
    # read original image to extract the dimension for the compressed image
    ori_image = cv2.imread(img)
    ori_image_rgb = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    img_array = np.array(ori_image_rgb, dtype='int32')

    # getting back the 3d matrix (row, col, rgb(3))
    row, col, rgb = img_array.shape

    # Get pixels
    pixels = read_image(img)

    # apply k-means algorithm
    labels, cluster_centers = k_means(pixels, num_colors)

    # Save the output into a file
    np.savetxt('.\data\output\cluster_labels.csv', labels, delimiter=',')
    np.savetxt('.\data\output\cluster_centers.csv', cluster_centers, delimiter=',')

    # assign pixels to its cluster centers
    compressed_img = cluster_centers[labels.astype(int), :]

    # reshape the compressed image to its original dimensions
    compressed_img_reshaped = np.reshape(compressed_img, (row, col, rgb), order="C")

    return compressed_img_reshaped


if __name__ == '__main__':

    # Images for testing
    beach = ".\data\beach.bmp"
    football = ".\data\football.bmp"
    sphynx = ".\data\sphynx.jpg"
    galaxy = ".\data\galaxy.jpg"
    stock_images = [beach, football, sphynx, galaxy]
    img_str = ['beach', 'football',  'Sphynx', 'galaxy']
    num_colors = [2, 4, 8, 16]
    img_idx = 0
    for img_path in stock_images:
        ori_image = cv2.imread(img_path)
        ori_image_rgb = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
        plt.imshow(ori_image_rgb)
        plt.title('Original ' + img_str[img_idx] + ' image')
        plt.show()
        for k in num_colors:
            compressed_image = compress_image(img_path, k)
            plt.imshow(compressed_image)
            plt.title('compressed image with ' + str(k) + ' colors', fontweight="bold")
            plt.savefig('.\data\output\compressed_with_' + str(k) + '_colors_' + img_str[img_idx] + '_image.png')
            plt.show()
        img_idx += 1
