import matplotlib.pyplot as plt
#x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
#y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
x = [2, 3, 5, 8, 9, 10, 12, 13, 15, 18, 19, 21, 22, 25, 27]
y = [10, 12, 14, 13, 15, 19, 20, 18, 21, 25, 27, 29, 26, 28, 30]
plt.scatter(x, y)
plt.show()
from sklearn.cluster import KMeans
data = list(zip(x, y))
inertias = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)
plt.plot(range(1,11), inertias, marker='X',color='r')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()
