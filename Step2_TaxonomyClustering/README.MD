# Step 2: Taxonomy Clustering
In the second step of our paper, we applied the KMeans clustering algorithm to group the 209 samples structured according to the taxonomy.

## Getting Started

### Dependencies

* Python 3.8

### Installing

* After downloading the repository, install the required packages by running the following command in the terminal:
``` 
pip install requirements -r requirements.txt
```

## Running the Topic Clustering
By executing the main.py script the kmeans algorithm is applied to the samples of the taxonomy using different number of clusters, identifying the optimal number of clusters using the silhouette score and the elbow method. The results of this can be found in the [results](results) and [plot](plot) folders.