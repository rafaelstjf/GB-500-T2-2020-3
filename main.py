import numpy as np  # multidimensional arrays
import math
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from sklearn.cluster import KMeans  # Kmeans algorithm from scikit-learning


def open_file():
    file_name = 'wine.data'
    header_name = 'wine_header.data'
    deli = ','
    data_elements = np.genfromtxt(file_name, delimiter=deli)
    data_header = np.genfromtxt(header_name, delimiter=deli, dtype=str)
    data_file = (data_elements, data_header)
    return data_file


def plot_all_raw_data(data_file):
    print(data_file[0])
    for i in range (1, len(data_file[1])):
        for j in range (i+1, len(data_file[1])):
            chart = plt.scatter(x=data_file[0][:, i], y=data_file[0][:, j], c=data_file[0][:, 0])
            plt.xlabel(data_file[1][i])
            plt.ylabel(data_file[1][j])
            plt.title("Raw data")
            plt.legend(*chart.legend_elements(), loc="upper right", title="Classes")
            plt.savefig("raw_data" + str(i) + "_" + str(j) + ".png")

def plot_raw_data(data_file, x_axys, y_axys):
    chart = plt.scatter(x=data_file[0][:, x_axys], y=data_file[0][:, y_axys], c=data_file[0][:, 0])
    plt.xlabel(data_file[1][x_axys])
    plt.ylabel(data_file[1][y_axys])
    plt.title("Raw data")
    plt.legend(*chart.legend_elements(), loc="upper right", title="Classes")
    plt.show()

def plot_result_data(data_file, result, x_axys, y_axys):
    chart = plt.scatter(x=data_file[0][:, x_axys], y=data_file[0][:, y_axys], c=result,)
    plt.xlabel(data_file[1][x_axys])
    plt.ylabel(data_file[1][y_axys])
    plt.title("Result")
    plt.legend(*chart.legend_elements(), loc="upper right", title="Classes")
    #plt.axes().add_artist(legend1)
    plt.show()

""" def plot_another_chart(data_file, result):
    a = []
    for i in range (0, len(result)):
        a.append(i)
    plt.plot(a, data_file[0][:,0],c='red')
    plt.scatter(a, result)
    plt.show() """
def run_kmeans(data_file, max_clusters=10):
    # creates the instance of kmeans object
    results = []
    wcss = []
    centers = []
    min_clusters = 2
    for i in range (min_clusters, max_clusters):
        k_means = KMeans(n_clusters=i)
        k_means.fit(data_file[0])  # Compute k-means clustering.
        wcss.append(k_means.inertia_)
        pred = k_means.predict(data_file[0])  # predict the clusters
        results.append(pred)
        centers.append(k_means.cluster_centers_)
    results = np.array(results)

    #elbow-method to choose the best cluster number
    x1, y1 = min_clusters, wcss[0]
    x2, y2 = max_clusters, wcss[len(wcss)-1]
    distances = []
    for i in range(len(wcss)):
        x0 = i+min_clusters
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    optimal_index = distances.index(max(distances))
    optimal_num =  optimal_index + min_clusters
    print("Optimal number of clusters: " + str(optimal_num))
    return results[optimal_index]


raw_data = open_file()
data = (raw_data[0][:, 1:], raw_data[1][1:])
plot_raw_data(raw_data, 1, 2)
result = run_kmeans(data, 8)
result+=1
plot_result_data(data, result, 0, 1)
# plot_another_chart(raw_data, result)
print(result)
