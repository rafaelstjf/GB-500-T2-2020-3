import numpy as np  # multidimensional arrays
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans  # Kmeans algorithm from scikit-learning
from sklearn import preprocessing
def open_file(data_file, header_file):
    file_name = data_file
    header_name = header_file
    deli = ','
    data_elements = np.genfromtxt(file_name, delimiter=deli)
    data_header = np.genfromtxt(header_name, delimiter=deli, dtype=str)
    return data_elements, data_header

def plot_raw_data(x_values, labels, headers, x_axys, y_axys):
    chart = plt.scatter(x=x_values[:, x_axys], y=x_values[:, y_axys], c=labels)
    plt.xlabel(headers[x_axys])
    plt.ylabel(headers[y_axys])
    plt.title("Agrupamentos originais")
    plt.legend(*chart.legend_elements(), loc="lower right", title="Agrupamentos")
    plt.show()

def count_result(result, labels):
    indexes={}
    for i in range(0, len(result)):
        if result[i] in indexes:
            indexes[result[i]].append(i)
        else:
            indexes[result[i]] = []
            indexes[result[i]].append(i)
    indexes = dict(sorted(indexes.items()))
    for key in indexes.keys():
        sum_set = 0
        sum_versi = 0
        sum_virg = 0
        for i in range(0, len(indexes[key])):
            if(labels[indexes[key][i]] == 0):
                sum_set+=1
            elif(labels[indexes[key][i]] == 1):
                sum_versi+=1
            elif(labels[indexes[key][i]] == 2):
                sum_virg+=1
        print('Cluster ' + str(key) + ' size: ' + str(len(indexes[key])))
        print('Setosa\'s frequency: ' + str((float)(sum_set/len(indexes[key]))))
        print('Versicolor\'s frequency: ' + str((float)(sum_versi/len(indexes[key]))))
        print('Virginica\'s frequency: ' + str((float)(sum_virg/len(indexes[key]))))
def plot_by_cluster(x_values, labels, headers, result, x_axys, y_axys):
    indexes ={}
    colors = []
    handles = []
    setosa = mpatches.Patch(color='#d54062', label='Iris-setosa')
    versicolor = mpatches.Patch(color='#ffa36c', label='Iris-versicolor')
    virginica = mpatches.Patch(color='#ebdc87', label='Iris-virginica')
    for i in range(0, len(labels)):
        colors.append('#eeeeee')
    for i in range(0, len(result)):
        if result[i] in indexes:
            indexes[result[i]].append(i)
        else:
            indexes[result[i]] = []
            indexes[result[i]].append(i)
    indexes = dict(sorted(indexes.items()))
    for key in indexes.keys():
        handles = []
        seto = False
        versi = False
        virg = False
        for c in range(0, len(colors)):
            colors[c] = '#eeeeee'
        for i in range(0, len(indexes[key])):
            ind = indexes[key][i];
            if(labels[ind] == 0):
                colors[ind] = '#d54062'
                seto = True
            elif(labels[ind] == 1):
                colors[ind] ='#ffa36c'
                versi = True
            elif(labels[ind] == 2):
                colors[ind] = '#ebdc87'
                virg = True
        if seto == True:
            handles.append(setosa)
        if versi == True:
            handles.append(versicolor)
        if virg == True:
            handles.append(virginica)
        chart = plt.scatter(x=x_values[:, x_axys], y=x_values[:, y_axys], c=colors)
        plt.xlabel(headers[x_axys])
        plt.ylabel(headers[y_axys])
        plt.title("Distribuição dos elementos no agrupamento " + str(key))
        plt.xlabel(headers[x_axys])
        plt.ylabel(headers[y_axys])
        plt.legend(handles=handles)
        plt.show()

            
            
def plot_result_data(x_values, headers, result, x_axys, y_axys):
    chart = plt.scatter(x=x_values[:, x_axys], y=x_values[:, y_axys], c=result)
    plt.xlabel(headers[x_axys])
    plt.ylabel(headers[y_axys])
    plt.title("Resultado do K-means")
    plt.show()

def run_kmeans(x, num_clusters):
    k_means = KMeans(n_clusters=num_clusters)
    k_means.fit(x)
    result = k_means.predict(x)
    return result
def run_kmeans_optimal(x, max_clusters=10):
    # creates the instance of kmeans object
    results = []
    wcss = []
    centers = []
    min_clusters = 2
    for i in range (min_clusters, max_clusters+1):
        k_means = KMeans(n_clusters=i)
        k_means.fit(x)  # Compute k-means clustering.
        wcss.append(k_means.inertia_)
        pred = k_means.predict(x)  # predict the clusters
        results.append(pred)
        centers.append(k_means.cluster_centers_)
    results = np.array(results)
    #elbow-method to choose the best cluster number
    optimal_index = 0
    optimal_num = 1
    if(len(wcss)>1):
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
    cl = []
    for i in range(min_clusters, (max_clusters+1)):
        cl.append(i)
    plt.plot(cl, wcss, color='#fddb3a')
    plt.plot([cl[0],cl[len(cl)-1]], [wcss[0],wcss[len(wcss)-1]], color='#f6f4e6')
    plt.plot(cl[optimal_index],wcss[optimal_index], color='#41444b', marker='x')
    plt.title('Método do Cotovelo')
    plt.ylabel('Distorção')
    plt.xlabel('Número de agrupamentos')
    plt.show()
    return results[optimal_index]

def main():
    data, headers = open_file('iris.data', 'iris_header.txt')
    #plot_all_raw_data(raw_data)
    x = data[:, :4]
    labels = data[:,4]
    #standard_x = preprocessing.scale(x)
    result = None
    run = True
    while(run == True):
        print('Options\n', 
        '\t1 - Run using 3 clusters\n',
        '\t2 - Run using elbow method (from 2 to 10 clusters)\n',
        '\t3 - Exit\n',
        )
        op = int(input('Type the option you want: '))
        if(op == 1):
            result = run_kmeans(x, 3)
        elif(op==2):
            result = run_kmeans_optimal(x, 10)
        elif(op==3):
            run = False
            return
        print("Dataset attributes:")
        for i in range(np.size(data[1], 0) - 1):
            print(str(i) + " - " + str(headers[i]))
        x_axis = int(input("which attribute do you want to see in the x-axis of the charts? "))
        y_axis = int(input("which attribute do you want to see in the y-axis of the charts? "))
        plot_raw_data(x, labels,headers, x_axis, y_axis)
        plot_result_data(x, headers, result, x_axis, y_axis)
        plot_by_cluster(x, labels, headers, result, x_axis, y_axis)
        count_result(result, labels)
main()