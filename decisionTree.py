import numpy as np  # multidimensional arrays
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import tree, preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix

def open_file(data_file, header_file):
    file_name = data_file
    header_name = header_file
    deli = ','
    data_elements = np.genfromtxt(file_name, delimiter=deli)
    data_header = np.genfromtxt(header_name, delimiter=deli, dtype=str)
    return data_elements, data_header
def build_matrix(y, result):
    indexes = {}
    for i in range(0, len(result)):
        if result[i] in indexes:
            indexes[result[i]].append(i)
        else:
            indexes[result[i]] = []
            indexes[result[i]].append(i)
    indexes = dict(sorted(indexes.items()))
    #fig, axs = plt.subplots(len(indexes.keys()), sharex=True)
    #fig.suptitle('Classificação das instâncias')
    #plt_ind = 0
    for key in indexes.keys():
        sum_correct = 0
        sum_wrong = 0
        for i in range(0, len(indexes[key])):
            ind = indexes[key][i]
            if(key == y[ind]):
                sum_correct+=1
            else:
                sum_wrong+=1
        print("Class " + str(key) + " :")
        print("\t Correctly classified: " + str(sum_correct))
        print("\t Wrongly classified: " + str(sum_wrong))
        '''
        axs[plt_ind].set_title("Classe " + str(key), fontsize=8, color='#838383')
        axs[plt_ind].bar(['Corretamente', 'Incorretamente'], [sum_correct, sum_wrong], width=0.1,color=['#799351','#d54062'])
        plt_ind+=1
    plt.show()
    '''

def run_decisionTree(x, y):
    clf = tree.DecisionTreeClassifier()
    sum_ = 0
    best_score = 0.0
    best_matrix = None
    best_result = []
    for i in range(0, 1000):
        clf.fit(x, y)
        result = clf.predict(x)
        curr_score = clf.score(x, y)
        if(curr_score > best_score):
            best_score = curr_score
            best_result = result
            best_matrix = confusion_matrix(y, result)
        sum_+=curr_score
    print('Average score:' + str(sum_/1000))
    print('Best score: ' + str(best_score))
    print(best_matrix)
    return best_result
    
def split_and_run_decisionTree(x, y, train_perc):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_perc, random_state=0, shuffle=True)
    clf = tree.DecisionTreeClassifier()
    sum_ = 0
    best_score = 0.0
    best_matrix = None
    best_result = []
    for i in range(0,1000):
        clf.fit(x_train, y_train)
        result = clf.predict(x_test)
        curr_score = clf.score(x_test, y_test)
        if(curr_score > best_score):
            best_score = curr_score
            best_result = result
            best_matrix = confusion_matrix(y_test, result)
        sum_+=curr_score
    print('Average score:' + str(sum_/1000))
    print('Best score: ' + str(best_score))
    print(best_matrix)
    return best_result, y_test

def cv_and_run_decisionTree(x, y, cv_value):
    #x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_perc, random_state=12, shuffle=True)
    parameters = {'max_depth':range(3,20)}
    clf = GridSearchCV(tree.DecisionTreeClassifier(), param_grid=parameters, n_jobs=4, cv=cv_value)
    clf.fit(X=x, y=y)
    tree_model = clf.best_estimator_
    print (clf.best_score_, clf.best_params_)
    result = clf.predict(x)
    allscores=clf.cv_results_['mean_test_score']
    print("All scores: ")
    print(allscores)
    print("Score: " + str(clf.score(x, y)))
    print(confusion_matrix(y, result))
    return result, y
def test_iris():
    data, headers = open_file('iris.data', 'iris_header.txt')
    x = data[:,:4]
    y = data[:,4]
    y_test = y
    #result = run_decisionTree(x, y)
    result, y_test = split_and_run_decisionTree(x, y, 0.2)
    #result, y_test = cv_and_run_decisionTree(x, y, 10)
    build_matrix(y_test, result)

def test_bcw():
    data, headers = open_file('wdbc.data', 'breast-cancer-wisconsin_header.txt')
    x = data [:,2:31]
    y = data[:,1]
    #imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    #x = imp.fit_transform(x)
    #caler = preprocessing.StandardScaler()
    scaler = preprocessing.MinMaxScaler()
    standard_x = scaler.fit_transform(x)
    result, y_test = split_and_run_decisionTree(standard_x, y, 0.2)
    #result, y_test = cv_and_run_decisionTree(standard_x, y, 10)
    #y_test = y
    #result = run_decisionTree(standard_x, y)
    build_matrix(y_test, result)


test_iris()