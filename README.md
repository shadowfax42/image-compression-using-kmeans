# Image compression using clustering

In this project, we will apply K-means algorithm to compress images. The algorithm is implemented from scratch instead of calling a package.

## Input
- `pixels`: the input image representation. Each row contains one data point (pixel). For image dataset, it contains 3 columns, each column corresponding to Red, Green, and Blue component. Each component has
an integer value between 0 and 255.

- `k`: the number of desired clusters. Too high value of K may result in empty cluster error. Then, you need to reduce it.

## Output
- `class`: cluster assignment of each data point in pixels. The assignment should be 1, 2, 3, etc. For k = 5, for example, each cell of class should be either 1, 2, 3, 4, or 5. The output should be a column vector with `size(pixels, 1)` elements. centroid: location of k centroids (or representatives) in your result. With images, each centroid corresponds to the representative color of each cluster. The output should be a matrix with K rows and 3 columns. The range of values should be [0, 255], possibly floating point numbers.
