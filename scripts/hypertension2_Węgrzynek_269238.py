# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 19:16:48 2022

@author: Jakub Węgrzynek 269238
"""

#%% Imports
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as pts

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay


from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)

#%% Set directories
project_dir = os.path.join(os.getcwd(), "hypertension_project_2")
data_dir = os.path.join(project_dir, "data")
os.makedirs(data_dir, exist_ok=True)
plot_dir = os.path.join(project_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

#%% Load data and split into training and test sets
tabela_pacjentow = pd.read_csv(os.path.join(data_dir,'Statystyka zbiorcza.csv'))
X = tabela_pacjentow.drop(columns=['Nazwa pliku', 'Obecność nadsiśnienia'])
Y = tabela_pacjentow['Obecność nadsiśnienia']

#%% Correlation matrix
plt.figure(figsize=(12,10))
cor = X.corr()
ax = sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
ax.xaxis.tick_top()
plt.xticks(rotation=90)
plt.savefig(os.path.join(plot_dir, "corrmatrix.jpg"), dpi = 300, bbox_inches='tight')
plt.show()

#%% Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#%% PCA

pca = PCA().fit(X_train)

plt.rcParams["figure.figsize"] = (12,6)

fig, ax = plt.subplots()
xi = np.arange(1, 15, step=1)
y = np.cumsum(pca.explained_variance_ratio_)


plt.ylim(0.0,1.1)
plt.xlim(0.0,15)
plt.plot(xi, y, marker='o', linestyle='--', color='b')

plt.xlabel('Number of Components')
plt.xticks(np.arange(0, 15, step=1)) #change from 0-based array index to 1-based human-readable label
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')

plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)

ax.grid(axis='x')
plt.show()

#%% K-fold Cross Validation x Times - function

def kfoldCVxTimes(names, models, X, Y, repeat, title, do_pca=False, pca_n_comp=1):

    test_accuracy = []
    test_precision = []
    test_recall = []
    test_f1 = []
    
    models_test_accuracy = []
    models_test_precision = []
    models_test_recall = []
    models_test_f1 = []

    test_accuracy_std = [[] for i in range(len(names))]
    test_precision_std = [[] for i in range(len(names))]
    test_recall_std = [[] for i in range(len(names))]
    test_f1_std = [[] for i in range(len(names))]    
    
    models_test_accuracy_std = []
    models_test_precision_std = []
    models_test_recall_std = []
    models_test_f1_std = []


    scoring = ['accuracy', 'precision', 'recall', 'f1']
    
    
    for rs in range(repeat):
        
        if do_pca == False:
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=rs)
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        
        if do_pca == True:
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=rs)
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            pca = PCA(n_components = pca_n_comp)
            pca.fit(X_train)
            X_train = pca.transform(X_train)
            X_test = pca.transform(X_test)
        
        for model, name in zip(models, names):
            CV = cross_validate(model, X_train, Y_train, scoring=scoring, cv=3)
            test_accuracy.append(np.mean(CV['test_accuracy']))
            test_precision.append(np.mean(CV['test_precision']))
            test_recall.append(np.mean(CV['test_recall']))
            test_f1.append(np.mean(CV['test_f1']))
            
            test_accuracy_std[names.index(name)].extend(CV['test_accuracy'])
            test_precision_std[names.index(name)].extend(CV['test_precision'])
            test_recall_std[names.index(name)].extend(CV['test_recall'])
            test_f1_std[names.index(name)].extend(CV['test_f1'])
            
    
    for i in range(len(names)):
        models_test_accuracy.append(np.mean(test_accuracy[i::len(names)]))
        models_test_precision.append(np.mean(test_precision[i::len(names)]))
        models_test_recall.append(np.mean(test_recall[i::len(names)]))
        models_test_f1.append(np.mean(test_f1[i::len(names)]))
    
        models_test_accuracy_std.append(np.std(test_accuracy_std[i]))
        models_test_precision_std.append(np.std(test_precision_std[i]))
        models_test_recall_std.append(np.std(test_recall_std[i]))
        models_test_f1_std.append(np.std(test_f1_std[i]))
        
        
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, facecolor = '#eaeaf2')
    fig.set_size_inches(14, 10)
    plt.subplots_adjust(hspace=0.6)
    fig.suptitle(title, fontsize=25)

    x = range(len(names))

    ax1.bar(x, models_test_accuracy)
    ax2.bar(x, models_test_precision)
    ax3.bar(x, models_test_recall)
    ax4.bar(x, models_test_f1)
    
    ax1.bar(x, models_test_accuracy_std)
    ax2.bar(x, models_test_precision_std)
    ax3.bar(x, models_test_recall_std)
    ax4.bar(x, models_test_f1_std)
    

    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 1)
    ax3.set_ylim(0, 1)
    ax4.set_ylim(0, 1)

    ax1.set_title("Accuracy test")
    ax2.set_title("Precision test")
    ax3.set_title("Recall test")
    ax4.set_title("F1 test")

    ax1.set_xticks(x)
    ax2.set_xticks(x)
    ax3.set_xticks(x)
    ax4.set_xticks(x)

    ax1.set_xticklabels(names, rotation=90)
    ax2.set_xticklabels(names, rotation=90)
    ax3.set_xticklabels(names, rotation=90)
    ax4.set_xticklabels(names, rotation=90)

    for i in range(len(x)):
        ax1.text(x[i]-0.3, models_test_accuracy[i]+0.01, str(round(models_test_accuracy[i],2)))
        ax2.text(x[i]-0.3, models_test_precision[i]+0.01, str(round(models_test_precision[i],2)))
        ax3.text(x[i]-0.3, models_test_recall[i]+0.01, str(round(models_test_recall[i],2)))
        ax4.text(x[i]-0.3, models_test_f1[i]+0.01, str(round(models_test_f1[i],2)))
        
        ax1.text(x[i]-0.3, models_test_accuracy_std[i]+0.02, str(round(models_test_accuracy_std[i],3)))
        ax2.text(x[i]-0.3, models_test_precision_std[i]+0.02, str(round(models_test_precision_std[i],3)))
        ax3.text(x[i]-0.3, models_test_recall_std[i]+0.02, str(round(models_test_recall_std[i],3)))
        ax4.text(x[i]-0.3, models_test_f1_std[i]+0.02, str(round(models_test_f1_std[i],3)))
        
    
    m = pts.Patch(color='#1f77b4', label="Mean")
    s = pts.Patch(color='orange', label="Std")
    ax1.legend(handles=[m,s], loc='best')
    ax2.legend(handles=[m,s], loc='best')
    ax3.legend(handles=[m,s], loc='best')
    ax4.legend(handles=[m,s], loc='best')
    plt.savefig(os.path.join(plot_dir, "{}.jpg".format(title)), dpi = 300, bbox_inches='tight')
    plt.show()
    

#%% Which classifier statistically performs the best?

names = [
    "SGD Classifier",
    "Logistic Regression",
    "Nearest Neighbors",
    "Linear SVC",
    "Decision Tree",
    "Random Forest",
    "Naive Bayes",
]

models = [
    SGDClassifier(random_state=42),
    LogisticRegression(max_iter=10000),
    KNeighborsClassifier(),
    SVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GaussianNB(),
]
print("All variables:")
kfoldCVxTimes(names, models, X, Y, repeat=60, title="All variables")

print("PCA(2):")
kfoldCVxTimes(names, models, X, Y, repeat=60, title="2 Components", do_pca=True, pca_n_comp=2)

print("PCA(8) - 95% of variance:")
kfoldCVxTimes(names, models, X, Y, repeat=60, title="8 Components - explained 95% of variance", do_pca=True, pca_n_comp=8)

#%% First conclusion
print("----> Wybrane klasyfikatory do dalszego badania:")
print('\n-LINEAR SVC i Gaussian Naive Bayes - wykazały się dużą wykrywalnością nadciśnienia przy \nstatystycznie niezbyt dużej ilości zdrowych zakwalifikowanych jako chorych oraz dużą stabilnością')
print('\n-SGD - wymaganie projektowe')

#%% PCA(2) Plot
pca = PCA(n_components = 2)
pca.fit(X_train)
X_train_pca2 = pca.transform(X_train)
X_test_pca2 = pca.transform(X_test)

sick = [X_train_pca2[i] for i in range(len(Y_train)) if list(Y_train)[i] == 1]
healthy = [X_train_pca2[i] for i in range(len(Y_train)) if list(Y_train)[i] == 0]

fig, ax = plt.subplots(facecolor = '#eaeaf2')
plt.scatter([l[0] for l in sick],[l[1] for l in sick],color='red', label = 'Hypertension')
plt.scatter([l[0] for l in healthy],[l[1] for l in healthy],color='blue', label = 'No Hypertension')
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_title('Plot of components for PCA(2)')
ax.legend(loc='best')
plt.show()


#%% GridSearch for Naive Bayes and PCA(2)

gnb = GaussianNB()
pipe1 = Pipeline(steps=[('gnb', gnb)])

var_smoothing = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8]
parameters = dict(gnb__var_smoothing=var_smoothing)

GNB_GS = GridSearchCV(pipe1, parameters, cv = 3)
GNB_GS.fit(X_train_pca2, Y_train)


print('Best var_smoothing:', GNB_GS.best_estimator_.get_params()['gnb__var_smoothing'])
print()

Y_test_predict = cross_val_predict(GNB_GS.best_estimator_, X_test_pca2, Y_test, cv = 3)

confusion =  confusion_matrix(Y_test, Y_test_predict)

print('Confusion matrix: \n', confusion, '\n')
print('TN = ', confusion[0,0], '\tFP = ', confusion[0,1])
print('FN = ', confusion[1,0], '\tTP = ', confusion[1,1])
print()
print('accuracy = %.3f'%accuracy_score(Y_test, Y_test_predict))
print('precision = %.3f'%(confusion[1,1]/(confusion[1,1] + confusion[0,1])) )
print('recall = %.3f'%(confusion[1,1]/(confusion[1,1] + confusion[1,0])) )
print()

#%% Grid Search for SVC and PCA(2)

svc = SVC()
pipe2 = Pipeline(steps=[('svc', svc)])

C = [1, 10, 100, 1000]
gamma = [1, 0.1, 0.01, 0.001]

parameters = dict(svc__C=C,
                  svc__gamma=gamma)

SVC_GS = GridSearchCV(pipe2, parameters, cv = 3)
SVC_GS.fit(X_train_pca2, Y_train)

print('Best C:', SVC_GS.best_estimator_.get_params()['svc__C'])
print('Best gamma:', SVC_GS.best_estimator_.get_params()['svc__gamma'])
print(); print(SVC_GS.best_estimator_.get_params()['svc'])

Y_test_predict = cross_val_predict(SVC_GS.best_estimator_, X_test_pca2, Y_test, cv = 3)

confusion =  confusion_matrix(Y_test, Y_test_predict)

print('Confusion matrix: \n', confusion, '\n')
print('TN = ', confusion[0,0], '\tFP = ', confusion[0,1])
print('FN = ', confusion[1,0], '\tTP = ', confusion[1,1])
print()
print('accuracy = %.3f'%accuracy_score(Y_test, Y_test_predict))
print('precision = %.3f'%(confusion[1,1]/(confusion[1,1] + confusion[0,1])) )
print('recall = %.3f'%(confusion[1,1]/(confusion[1,1] + confusion[1,0])) )
print()


#%% Grid Search for SGD and PCA(2)

sgd = SGDClassifier(random_state=42)
pipe3 = Pipeline(steps=[('sgd', sgd)])

alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
loss = ['hinge', 'log', 'squared_hinge']
penalty = ['l2', 'l1', 'elasticnet']
eta0 = [0.001, 0.01, 0.1, 1, 10, 100]
    
parameters = dict(sgd__alpha=alpha,
                  sgd__loss=loss,
                  sgd__penalty=penalty,
                  sgd__eta0=eta0)

SGD_GS = GridSearchCV(pipe3, parameters, cv = 3)
SGD_GS.fit(X_train_pca2, Y_train)

print('Best alpha:', SGD_GS.best_estimator_.get_params()['sgd__alpha'])
print('Best loss:', SGD_GS.best_estimator_.get_params()['sgd__loss'])
print('Best penalty:', SGD_GS.best_estimator_.get_params()['sgd__penalty'])
print('Best eta0:', SGD_GS.best_estimator_.get_params()['sgd__eta0'])
print(); print(SGD_GS.best_estimator_.get_params()['sgd'])

Y_test_predict = cross_val_predict(SGD_GS.best_estimator_, X_test_pca2, Y_test, cv = 3)

confusion =  confusion_matrix(Y_test, Y_test_predict)

print('Confusion matrix: \n', confusion, '\n')
print('TN = ', confusion[0,0], '\tFP = ', confusion[0,1])
print('FN = ', confusion[1,0], '\tTP = ', confusion[1,1])
print()
print('accuracy = %.3f'%accuracy_score(Y_test, Y_test_predict))
print('precision = %.3f'%(confusion[1,1]/(confusion[1,1] + confusion[0,1])) )
print('recall = %.3f'%(confusion[1,1]/(confusion[1,1] + confusion[1,0])) )
print()

#%% Lets check mean performance of our models after GridSearch and with PCA(2)
names = [
    "SGD Classifier",
    "Naive Bayes",
    "Linear SVC"
]

models = [
    SGDClassifier(alpha=0.1, eta0=0.001, loss='hinge', penalty= 'l2', random_state=42),
    GaussianNB(var_smoothing=1e-12),
    SVC(C=100, gamma=0.001)
]

print("After reducing to 2 components:")
kfoldCVxTimes(names, models, X_train, Y_train, 60, title="Hyperparameters and 2 components", do_pca=True, pca_n_comp=2)

#%% Data PCA(8)

pca = PCA(8)
pca.fit(X_train)
X_train_pca8 = pca.transform(X_train)
X_test_pca8 = pca.transform(X_test)

#%% GridSearch for Naive Bayes and PCA(8)

gnb = GaussianNB()
pipe1 = Pipeline(steps=[('gnb', gnb)])

var_smoothing = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8]
parameters = dict(gnb__var_smoothing=var_smoothing)

GNB_GS = GridSearchCV(pipe1, parameters, cv = 3)
GNB_GS.fit(X_train_pca8, Y_train)


print('Best var_smoothing:', GNB_GS.best_estimator_.get_params()['gnb__var_smoothing'])
print()

Y_test_predict = cross_val_predict(GNB_GS.best_estimator_, X_test_pca8, Y_test, cv = 3)

confusion =  confusion_matrix(Y_test, Y_test_predict)

print('Confusion matrix: \n', confusion, '\n')
print('TN = ', confusion[0,0], '\tFP = ', confusion[0,1])
print('FN = ', confusion[1,0], '\tTP = ', confusion[1,1])
print()
print('accuracy = %.3f'%accuracy_score(Y_test, Y_test_predict))
print('precision = %.3f'%(confusion[1,1]/(confusion[1,1] + confusion[0,1])) )
print('recall = %.3f'%(confusion[1,1]/(confusion[1,1] + confusion[1,0])) )
print()
#%% Grid Search for SVC and PCA(8)

svc = SVC()
pipe2 = Pipeline(steps=[('svc', svc)])

C = [1, 10, 100, 1000]
gamma = [1, 0.1, 0.01, 0.001]

parameters = dict(svc__C=C,
                  svc__gamma=gamma)

SVC_GS = GridSearchCV(pipe2, parameters, cv = 3)
SVC_GS.fit(X_train_pca8, Y_train)

print('Best C:', SVC_GS.best_estimator_.get_params()['svc__C'])
print('Best gamma:', SVC_GS.best_estimator_.get_params()['svc__gamma'])
print(); print(SVC_GS.best_estimator_.get_params()['svc'])

Y_test_predict = cross_val_predict(SVC_GS.best_estimator_, X_test_pca8, Y_test, cv = 3)

confusion =  confusion_matrix(Y_test, Y_test_predict)

print('Confusion matrix: \n', confusion, '\n')
print('TN = ', confusion[0,0], '\tFP = ', confusion[0,1])
print('FN = ', confusion[1,0], '\tTP = ', confusion[1,1])
print()
print('accuracy = %.3f'%accuracy_score(Y_test, Y_test_predict))
print('precision = %.3f'%(confusion[1,1]/(confusion[1,1] + confusion[0,1])) )
print('recall = %.3f'%(confusion[1,1]/(confusion[1,1] + confusion[1,0])) )
print()


#%% Grid Search for SGD and PCA(8)

sgd = SGDClassifier(random_state=42)
pipe3 = Pipeline(steps=[('sgd', sgd)])

alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
loss = ['hinge', 'log', 'squared_hinge']
penalty = ['l2', 'l1', 'elasticnet']
eta0 = [0.001, 0.01, 0.1, 1, 10, 100]
    
parameters = dict(sgd__alpha=alpha,
                  sgd__loss=loss,
                  sgd__penalty=penalty,
                  sgd__eta0=eta0)

SGD_GS = GridSearchCV(pipe3, parameters, cv = 3)
SGD_GS.fit(X_train_pca8, Y_train)

print('Best alpha:', SGD_GS.best_estimator_.get_params()['sgd__alpha'])
print('Best loss:', SGD_GS.best_estimator_.get_params()['sgd__loss'])
print('Best penalty:', SGD_GS.best_estimator_.get_params()['sgd__penalty'])
print('Best eta0:', SGD_GS.best_estimator_.get_params()['sgd__eta0'])
print(); print(SGD_GS.best_estimator_.get_params()['sgd'])

Y_test_predict = cross_val_predict(SGD_GS.best_estimator_, X_test_pca8, Y_test, cv = 3)

confusion =  confusion_matrix(Y_test, Y_test_predict)

print('Confusion matrix: \n', confusion, '\n')
print('TN = ', confusion[0,0], '\tFP = ', confusion[0,1])
print('FN = ', confusion[1,0], '\tTP = ', confusion[1,1])
print()
print('accuracy = %.3f'%accuracy_score(Y_test, Y_test_predict))
print('precision = %.3f'%(confusion[1,1]/(confusion[1,1] + confusion[0,1])) )
print('recall = %.3f'%(confusion[1,1]/(confusion[1,1] + confusion[1,0])) )
print()


#%% Lets check mean performance of our models after GridSearch and with PCA(8)
names = [
    "SGD Classifier",
    "Naive Bayes",
    "Linear SVC"
]

models = [
    SGDClassifier(alpha=1, eta0=0.001, loss='log', penalty= 'l2', random_state=42),
    GaussianNB(var_smoothing=1e-12),
    SVC(C=10, gamma=0.01)
]

print("After reducing to 8 components")
kfoldCVxTimes(names, models, X_train, Y_train, 60, title="Hyperparameters and 8 components", do_pca=True, pca_n_comp=8)
print("After reducing to 8 components we got better results")


#%% Grid Search for Naive Bayes - best hyperparameters and all components

gnb = GaussianNB()
pipe1 = Pipeline(steps=[('gnb', gnb)])

var_smoothing = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8]

parameters = dict(gnb__var_smoothing=var_smoothing)

GNB_GS = GridSearchCV(pipe1, parameters, cv = 3)
GNB_GS.fit(X_train, Y_train)


print('Best var_smoothing:', GNB_GS.best_estimator_.get_params()['gnb__var_smoothing'])
print()

Y_test_predict = cross_val_predict(GNB_GS.best_estimator_, X_test, Y_test, cv = 3)

confusion =  confusion_matrix(Y_test, Y_test_predict)

print('Confusion matrix: \n', confusion, '\n')
print('TN = ', confusion[0,0], '\tFP = ', confusion[0,1])
print('FN = ', confusion[1,0], '\tTP = ', confusion[1,1])
print()
print('accuracy = %.3f'%accuracy_score(Y_test, Y_test_predict))
print('precision = %.3f'%(confusion[1,1]/(confusion[1,1] + confusion[0,1])) )
print('recall = %.3f'%(confusion[1,1]/(confusion[1,1] + confusion[1,0])) )
print()


#%% Grid Search for SVC - best hyperparameters and all components

svc = SVC()
pipe2 = Pipeline(steps=[('svc', svc)])

C = [1, 10, 100, 1000]
gamma = [1, 0.1, 0.01, 0.001]

parameters = dict(svc__C=C,
                  svc__gamma=gamma)

SVC_GS = GridSearchCV(pipe2, parameters, cv = 3)
SVC_GS.fit(X_train, Y_train)

print('Best C:', SVC_GS.best_estimator_.get_params()['svc__C'])
print('Best gamma:', SVC_GS.best_estimator_.get_params()['svc__gamma'])
print(); print(SVC_GS.best_estimator_.get_params()['svc'])

Y_test_predict = cross_val_predict(SVC_GS.best_estimator_, X_test, Y_test, cv = 3)

confusion =  confusion_matrix(Y_test, Y_test_predict)

print('Confusion matrix: \n', confusion, '\n')
print('TN = ', confusion[0,0], '\tFP = ', confusion[0,1])
print('FN = ', confusion[1,0], '\tTP = ', confusion[1,1])
print()
print('accuracy = %.3f'%accuracy_score(Y_test, Y_test_predict))
print('precision = %.3f'%(confusion[1,1]/(confusion[1,1] + confusion[0,1])) )
print('recall = %.3f'%(confusion[1,1]/(confusion[1,1] + confusion[1,0])) )
print()


#%% Grid Search for SGD - best hyperparameters and number of components

sgd = SGDClassifier(random_state=42)
pipe3 = Pipeline(steps=[('sgd', sgd)])

alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
loss = ['hinge', 'log', 'squared_hinge']
penalty = ['l2', 'l1', 'elasticnet']
eta0 = [0.001, 0.01, 0.1, 1, 10, 100]
    
parameters = dict(sgd__alpha=alpha,
                  sgd__loss=loss,
                  sgd__penalty=penalty,
                  sgd__eta0=eta0)

SGD_GS = GridSearchCV(pipe3, parameters, cv = 3)
SGD_GS.fit(X_train, Y_train)

print('Best alpha:', SGD_GS.best_estimator_.get_params()['sgd__alpha'])
print('Best loss:', SGD_GS.best_estimator_.get_params()['sgd__loss'])
print('Best penalty:', SGD_GS.best_estimator_.get_params()['sgd__penalty'])
print('Best eta0:', SGD_GS.best_estimator_.get_params()['sgd__eta0'])
print(); print(SGD_GS.best_estimator_.get_params()['sgd'])

Y_test_predict = cross_val_predict(SGD_GS.best_estimator_, X_test, Y_test, cv = 3)

confusion =  confusion_matrix(Y_test, Y_test_predict)

print('Confusion matrix: \n', confusion, '\n')
print('TN = ', confusion[0,0], '\tFP = ', confusion[0,1])
print('FN = ', confusion[1,0], '\tTP = ', confusion[1,1])
print()
print('accuracy = %.3f'%accuracy_score(Y_test, Y_test_predict))
print('precision = %.3f'%(confusion[1,1]/(confusion[1,1] + confusion[0,1])) )
print('recall = %.3f'%(confusion[1,1]/(confusion[1,1] + confusion[1,0])) )
print()

#%% Lets check mean performance of our models after GridSearch and with all components
names = [
    "SGD Classifier",
    "Naive Bayes",
    "Linear SVC"
]

models = [
    SGDClassifier(alpha=0.1, eta0=0.001, loss='hinge', penalty= 'l1', random_state=42),
    GaussianNB(var_smoothing=1e-12),
    SVC(C=10, gamma=0.01)
]

kfoldCVxTimes(names, models, X_train, Y_train, 60, title="Hyperparameters and all variables")

#%%

print("---> SGD Classifier statistically did the best job with tuned hyperparameters and PCA(2)")
print("---> Naive bayes statistically did the best job with tuned hyperparameters and PCA(8)")
print("---> SVC statistically did the best job with tuned hyperparameters and with given all variables")
print()
print("But it's extremely hard to judge which model is the best because we have not enough data ")


#%% Plots and final prediction for final SGD with PCA(2)

final_sgd= SGDClassifier(alpha=0.1, eta0=0.001, loss='hinge', penalty= 'l2', random_state=42)
final_sgd.fit(X_test_pca2, Y_test)
final_predictions = final_sgd.predict(X_test_pca2)

final_sgd_accuracy = accuracy_score(Y_test,final_predictions)
final_sgd_f1 = f1_score(Y_test,final_predictions)
final_sgd_recall = recall_score(Y_test,final_predictions)
final_sgd_precision = precision_score(Y_test,final_predictions)

print('Accuracy of the best SGDClassifier model =',"%.2f"%final_sgd_accuracy)
print('f1 of the best SGDClassifier model =',"%.2f"%final_sgd_f1)
print('Recall of the best SGDClassifier model =',"%.2f"%final_sgd_recall)
print('Presision of the best SGDClassifier model =',"%.2f"%final_sgd_precision)


display = PrecisionRecallDisplay.from_estimator(
    final_sgd, X_test_pca2, Y_test, name="SGD"
)
_ = display.ax_.set_title("Precision vs Recall")


#%% Plots and final prediction for final Naive Bayes with PCA(8)

final_nb = GaussianNB(var_smoothing=1e-12)
final_nb.fit(X_test_pca8, Y_test)
final_nb_predictions = final_nb.predict(X_test_pca8)

final_nb_accuracy = accuracy_score(Y_test,final_nb_predictions)
final_nb_f1 = f1_score(Y_test,final_nb_predictions)
final_nb_recall = recall_score(Y_test,final_nb_predictions)
final_nb_precision = precision_score(Y_test,final_nb_predictions)

print('Accuracy of the best Naive Bayes model =',"%.2f"%final_nb_accuracy)
print('f1 of the best Nauve Bayes model =',"%.2f"%final_nb_f1)
print('Recall of the best Naive Bayes model =',"%.2f"%final_nb_recall)
print('Presision of the best Naive Bayes model =',"%.2f"%final_nb_precision)

display = PrecisionRecallDisplay.from_estimator(
    final_nb, X_test_pca8, Y_test, name="Naive Bayes"
)
_ = display.ax_.set_title("Precision vs Recall")


#%% Plots and final prediction for final SVC with all variables

final_svc = SVC(C=10, gamma=0.01)
final_svc.fit(X_test, Y_test)
final_svc_predictions = final_svc.predict(X_test)

final_svc_accuracy = accuracy_score(Y_test,final_svc_predictions)
final_svc_f1 = f1_score(Y_test,final_svc_predictions)
final_svc_recall = recall_score(Y_test,final_svc_predictions)
final_svc_precision = precision_score(Y_test,final_svc_predictions)

print('Accuracy of the best SVC model =',"%.2f"%final_svc_accuracy)
print('f1 of the best SVC model =',"%.2f"%final_svc_f1)
print('Recall of the best SVC model =',"%.2f"%final_svc_recall)
print('Presision of the best SVC model =',"%.2f"%final_svc_precision)

display = PrecisionRecallDisplay.from_estimator(
    final_svc, X_test, Y_test, name="SVC"
)
_ = display.ax_.set_title("Precision vs Recall")













