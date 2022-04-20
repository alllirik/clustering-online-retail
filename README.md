# Clustering techniques for online retail
A small research project regarding different ways to cluster users in online retail using binary vectors of items bought by each user.

This project uses two datasets:
1. https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset?select=events.csv
2. https://www.kaggle.com/retailrocket/ecommerce-dataset?select=item_properties_part1.csv

###  k-means with svd for online-retail.ipynb
This notebook contains some analytical insights into the retail dataset, dimensionality reduction using SVD, clustering using k-means, and visualization.

### clustering for online-retail using k-means and pca.ipynb
This notebook contains comparison of PCA clusterization vs. SVD+K-means clusterization.

### k-c-cluster.py
This module containts several classes that contain methods of clustering (K-means, K-medoids, C-means) users represented by binary vectors.
