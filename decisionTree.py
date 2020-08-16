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
def run_decisionTree(x, y):
    clf = tree.DecisionTreeClassifier(random_state=0)
    clf.fit(x, y)
    result = clf.predict(x)
    score = clf.score(x, y)
    matrix = confusion_matrix(y, result)
    print('Score: ' + str(score))
    print('Confusion Matrix:')
    print(matrix)
    return result
    
def split_and_run_decisionTree(x, y, train_perc):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_perc, random_state=0, shuffle=True)
    clf = tree.DecisionTreeClassifier(random_state=0)
    clf.fit(x_train, y_train)
    result = clf.predict(x_test)
    score = clf.score(x_test, y_test)
    matrix = confusion_matrix(y_test, result)
    print('Score: ' + str(score))
    print(matrix)
    return result, y_test

def cv_and_run_decisionTree(x, y, cv_value, names, class_names):
    #x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_perc, random_state=12, shuffle=True)
    parameters = {'max_depth':range(3,20)}
    clf = GridSearchCV(tree.DecisionTreeClassifier(random_state=0), param_grid=parameters, n_jobs=4, cv=cv_value)
    clf.fit(X=x, y=y)
    print('Status')
    for i in clf.cv_results_:
        print(i, '-> ',clf.cv_results_[i])
    sum_=0
    l = list(clf.cv_results_['std_test_score'])
    for i in range(0,len(l)):
        sum_+=l[i]
    print('Standard deviatation: ', end='')
    print(sum_/len(l))
    tree_model = clf.best_estimator_
    print (clf.best_score_, clf.best_params_)
    result = clf.predict(x)
    return result, y

def cv_and_split_and_run_decisionTree(x, y, cv_value, names, class_names, train_perc):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_perc, random_state=12, shuffle=True)
    parameters = {'max_depth':range(3,20)}
    clf = GridSearchCV(tree.DecisionTreeClassifier(random_state=0), param_grid=parameters, n_jobs=4, cv=cv_value)
    clf.fit(X=x_train, y=y_train)
    print('Status')
    for i in clf.cv_results_:
        print(i, '-> ',clf.cv_results_[i])
    sum_=0
    l = list(clf.cv_results_['std_test_score'])
    for i in range(0,len(l)):
        sum_+=l[i]
    print('Standard deviatation: ', end='')
    print(sum_/len(l))
    tree_model = clf.best_estimator_
    #fig = plt.figure(dpi=1200)
    #tree.plot_tree(tree_model, feature_names=names, class_names=class_names, label='root', rounded=True)
    #fig.savefig('tree.png', dpi=fig.dpi)
    print (clf.best_score_, clf.best_params_)
    result = clf.predict(x)
    print("Score: " + str(clf.score(x_test, y_test)))
    print(confusion_matrix(y, result))
    return result, y

def test_iris():
    data, headers = open_file('iris.data', 'iris_header.txt')
    x = data[:,:4]
    y = data[:,4]
    names=headers[:4]
    class_names=['setosa', 'versicolour', 'virginica']
    y_test = y
    #result = run_decisionTree(x, y)
    #result, y_test = split_and_run_decisionTree(x, y, 0.8)
    result, y_test = cv_and_run_decisionTree(x, y, 10, names,class_names)
    #build_matrix(y_test, result)

def test_bcw():
    data, headers = open_file('wdbc.data', 'breast-cancer-wisconsin_header.txt')
    x = data [:,2:31]
    y = data[:,1]
    names=headers[2:31]
    class_names = ['M', 'B']
    #imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    #x = imp.fit_transform(x)
    #caler = preprocessing.StandardScaler()
    scaler = preprocessing.MinMaxScaler()
    standard_x = scaler.fit_transform(x)
    #result, y_test = split_and_run_decisionTree(standard_x, y, 0.2)
    #result, y_test = cv_and_run_decisionTree(standard_x, y, 10, names, class_names)
    result, y_test = cv_and_split_and_run_decisionTree(standard_x, y, 10, names, class_names, 0.8)
    
    #y_test = y
    #result = run_decisionTree(standard_x, y)

test_bcw()