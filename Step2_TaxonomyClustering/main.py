import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

random_state = 42

# load data
data_raw = pd.read_excel("./data/ATaxconomyForDataScarcitySolutions.xlsx", sheet_name="V11", dtype=str)
data = data_raw.fillna("None")
temp = pd.DataFrame()
for column in data.columns[7:21]:
    dummies = data[column].str.get_dummies(sep=", ")
    dummies.columns = '{}-'.format(column) + dummies.columns
    temp = pd.concat((temp, dummies), axis= 1)
data = pd.concat((data.iloc[:, :7], temp), axis = 1)

# apply kmeans clustering with sihouette score
range_n_clusters = list(range(3,15))
fig, axs = plt.subplots(6, 2, figsize=(15, 10))

for idx, n_clusters in enumerate(range_n_clusters):
    ax1 = axs[idx//2, idx%2]
    clusterer = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = clusterer.fit_predict(data.iloc[:, 7:]) + 1

    silhouette_avg = silhouette_score(data.iloc[:, 7:], cluster_labels)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

    sample_silhouette_values = silhouette_samples(data.iloc[:, 7:], cluster_labels)

    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(data.iloc[:, 7:]) + (n_clusters + 1) * 10])

    y_lower = 10
    for i in range(1, n_clusters+1):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = sns.cubehelix_palette(n_clusters)[i-1]
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), rotation='vertical')

        y_lower = y_upper + 10

    ax1.set_title(f"Silhouette plot for k = {n_clusters}")
    ax1.set_xlabel("Silhouette coefficient")
    ax1.set_ylabel("Cluster")

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    cluster_labels = clusterer.fit_predict(data.iloc[:, 7:]) + 1
    data_raw['Cluster'] = cluster_labels
    data_raw.to_excel('./results/ATaxconomyForDataScarcitySolutions_numClust_{}.xlsx'.format(n_clusters), index=False)

plt.tight_layout()
plt.savefig('./plots/comparison_silhouette_score.png', dpi=600)
plt.show()

# elbow method
max_clusters = 15

wcss = []
dist = []
for i in range(1, max_clusters+1):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data.iloc[:, 7:])
    wcss.append(kmeans.inertia_)
    dist.append(sum(np.min(cdist(data.iloc[:, 7:], kmeans.cluster_centers_, 'euclidean'), axis=1)) / data.iloc[:, 7:].shape[0])

var_explained = [1 - (x/wcss[0]) for x in wcss]

plt.figure(figsize=(10, 7))
plt.plot(range(1, max_clusters+1), var_explained, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Variance Explained')
plt.title('Elbow Method ')
plt.savefig('./plots/comparison_elbow_varExp_score.png', dpi=600)
plt.show()

plt.figure(figsize=(10, 7))
plt.plot(range(1, max_clusters+1), wcss, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method ')
plt.savefig('./plots/comparison_elbow_varExp_score.png', dpi=600)
plt.show()

plt.figure(figsize=(10, 7))
plt.plot(range(1, max_clusters+1), dist, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortions')
plt.title('Elbow Method ')
plt.savefig('./plots/comparison_elbow_dist_score.png', dpi=600)
plt.show()



