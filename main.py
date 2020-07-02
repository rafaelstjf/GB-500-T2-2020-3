import numpy as np  # multidimensional arrays
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
    print(data_file[0])
    chart = plt.scatter(x=data_file[0][:, x_axys], y=data_file[0][:, y_axys], c=data_file[0][:, 0])
    plt.xlabel(data_file[1][x_axys])
    plt.ylabel(data_file[1][y_axys])
    plt.title("Raw data")
    plt.legend(*chart.legend_elements(), loc="upper right", title="Classes")
    plt.show()

def plot_result_data(data_file, result, x_axys, y_axys):
    print(data_file[0])
    chart = plt.scatter(x=data_file[0][:, x_axys], y=data_file[0][:, y_axys], c=result,)
    plt.xlabel(data_file[1][x_axys])
    plt.ylabel(data_file[1][y_axys])
    plt.title("Result")
    plt.legend(*chart.legend_elements(), loc="upper right", title="Classes")
    #plt.axes().add_artist(legend1)
    plt.show()

def plot_another_chart(data_file, result):
    a = []
    for i in range (0, len(result)):
        a.append(i)
    plt.plot(a, data_file[0][:,0],c='red')
    plt.scatter(a, result)
    plt.show()

def run_kmeans(data_file, num_clusters=10):
    # creates the instance of kmeans object
    k_means = KMeans(n_clusters=num_clusters)
    k_means.fit(data_file[0])  # Compute k-means clustering.
    pred = k_means.predict(data_file[0])  # predict the clusters
    return pred


raw_data = open_file()
data = (raw_data[0][:, 1:], raw_data[1][1:])
plot_raw_data(raw_data, 1, 2)
result = run_kmeans(data, 3)
result+=1
plot_result_data(data, result, 0, 1)
plot_another_chart(raw_data, result)
