import numpy as np  # multidimensional arrays
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans  # Kmeans algorithm from scikit-learning

def open_file():
    file_name = 'seeds_dataset.csv'
    header_name = 'seeds_dataset_header.csv'
    deli = ','
    data_elements = np.genfromtxt(file_name, delimiter=deli)
    data_header = np.genfromtxt(header_name, delimiter=deli, dtype=str)
    data_file = (data_elements, data_header)
    return data_file

def plot_raw_data(data_file, x_axys, y_axys):
    chart = plt.scatter(x=data_file[0][:, x_axys], y=data_file[0][:, y_axys], c=data_file[0][:, 7])
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
    plt.show()
def standardisation(data_file):
    # scikit can do it using
    # from sklearn import preprocessing
    # preprocessing.scale(data_file[0])
    # but i've already done it so =)
    average = []
    for i in range(np.size(data_file[0], 1)):
        average.append(np.sum(data_file[0][:, i]))
    for i in range(np.size(data_file[0], 1)):
        average[i] /= np.size(data_file[0], 0)
    standard_dev = []
    for i in range(np.size(data_file[0], 1)):
        sum = 0
        for j in range(np.size(data_file[0], 0)):
            sum+= ((data_file[0][j, i])-average[i])**2
        standard_dev.append(math.sqrt((sum/np.size(data_file[0], 0))))
    fixed = data_file
    for i in range(np.size(data_file[0], 1)):
        for j in range(np.size(data_file[0], 0)):
            fixed[0][j, i] = (data_file[0][j, i]-average[i])/standard_dev[i]
    return fixed
    
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

def main():
    raw_data = open_file()
    #plot_all_raw_data(raw_data)
    data = (raw_data[0][:, :7], raw_data[1][:7]) # the seeds dataset's last column is the class
    plot_raw_data(raw_data, 0, 4)
    standard_data = standardisation(data)
    result = run_kmeans(standard_data, 8)
    result+=1
    plot_result_data(data, result, 0, 1)
    #print(result)
main()