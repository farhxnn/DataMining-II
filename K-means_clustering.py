# Step 1: Import libraries
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 2: Sample data
x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
data = list(zip(x, y))  # Combine x and y into coordinate pairs

# Step 3: Visualize the data
plt.scatter(x, y)
plt.title("Original Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Step 4: Find the best number of clusters using the Elbow method
inertias = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

# Step 5: Plot the Elbow graph
plt.plot(range(1, 11), inertias, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()

