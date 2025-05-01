import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
# Generate some example data
np.random.seed(0)
data = np.random.rand(10, 2) # 10 random data points with 2 features
# Perform hierarchical clustering
linked = linkage(data, method='average') # You can change 'average' to 'single' or 'complete'
# Create a dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data points')
plt.ylabel('Distance')
plt.show()
