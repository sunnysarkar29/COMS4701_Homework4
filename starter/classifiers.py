from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from matplotlib import colormaps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

class Classifiers():
    def __init__(self,data):
        '''
        TODO: Write code to convert the given pandas dataframe into training and testing data
        # all the data should be nxd arrays where n is the number of samples and d is the dimension of the data
        # all the labels should be nx1 vectors with binary labels in each entry
        '''

        figPartA, axPartA = plt.subplots()

        for _a, _b, _label in zip(data.A, data.B, data.label):
            axPartA.scatter(_a, _b, color=('red' if _label == 1 else 'blue'), \
                                    marker='o' if _label == 1 else 'x', \
                                    label='Classifier 1' if _label == 1 else 'Classifier 0')

        plt.title('Part A: Data Visualization')
        plt.xlabel('A')
        plt.ylabel('B')
        plt.savefig('partA.png')
        plt.show()
        # plt.show(block=False)

        data_numpy = df.to_numpy()
        X = data_numpy[:, :2]
        y = data_numpy[:, 2]

        self.training_data, self.testing_data, self.training_labels, self.testing_labels = train_test_split(X, y, test_size=0.4, random_state=1)


        orig_stdout = sys.stdout
        f = open('partB.txt', 'w')
        sys.stdout = f

        print(f'Training Data: [Number of samples: {len(self.training_labels)}]')
        print(self.training_data)
        print('\nTraining Labels: ')
        print(self.training_labels)
        print(f'\nTesting Data: [Number of samples: {len(self.testing_labels)}]')
        print(self.testing_data)
        print('\nTesting Labels: ')
        print(self.testing_labels)

        sys.stdout = orig_stdout
        f.close()

        self.foldDatas, self.foldLabels = self.get5FoldSplit(self.training_data, self.training_labels)

        self.outputs = []

    def get5FoldSplit(self, data, labels, num_folds=5):
        len_data = len(data)
        fold_size = len_data // num_folds
        foldsD, foldsL = [], []
        idx = [i * fold_size for i in range(num_folds)]

        for i in range(num_folds):
            if i == num_folds - 1:
                d = data[idx[i]:]
                l = labels[idx[i]:]
            else:
                d = data[idx[i]:idx[i+1]]
                l = labels[idx[i]:idx[i+1]]

            foldsD.append(d)
            foldsL.append(l)

        return foldsD, foldsL


    def test_clf(self, clf, classifier_name=''):
        # TODO: Fit the classifier and extrach the best score, training score and parameters
        clf.fit(self.training_data, self.training_labels)

        trainingScore = clf.best_score_

        df = pd.DataFrame(clf.cv_results_)
        df.to_csv(f'{classifier_name}.csv', index=False)

        clf = clf.best_estimator_
        testingScore = clf.score(self.testing_data, self.testing_labels)

        self.outputs.append(f'{classifier_name}, {trainingScore}, {testingScore}')
        self.plot(self.testing_data, clf.predict(self.testing_data),model=clf,classifier_name=classifier_name)

    def classifyNearestNeighbors(self):
        # TODO: Write code to run a Nearest Neighbors classifier
        clf = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': range(1, 20, 2), 'leaf_size': range(5, 31, 5)}, cv=None)
        self.test_clf(clf, 'NearestNeighbors')

    def classifyLogisticRegression(self):
        # TODO: Write code to run a Logistic Regression classifier
        clf = GridSearchCV(LogisticRegression(), {'C': [0.1, 0.5, 1, 5 ,10, 50, 100]}, cv=None)
        self.test_clf(clf, 'LogisticRegression')

    def classifyDecisionTree(self):
        clf = GridSearchCV(DecisionTreeClassifier(), {'max_depth': range(1, 51), 'min_samples_split': range(2, 11)}, cv=None)
        self.test_clf(clf, 'DecissionTree')

    def classifyRandomForest(self):
        clf = GridSearchCV(RandomForestClassifier(), {'max_depth': range(1, 6), 'min_samples_split': range(2, 11)}, cv=None)
        self.test_clf(clf, 'RandomForest')

    def classifyAdaBoost(self):
        # TODO: Write code to run a AdaBoost classifier
        clf = GridSearchCV(AdaBoostClassifier(), {'n_estimators': range(10, 71, 10)}, cv=None)
        self.test_clf(clf, 'AdaBoost')

    def plot(self, X, Y, model,classifier_name = ''):
        X1 = X[:, 0]
        X2 = X[:, 1]

        X1_min, X1_max = min(X1) - 0.5, max(X1) + 0.5
        X2_min, X2_max = min(X2) - 0.5, max(X2) + 0.5

        X1_inc = (X1_max - X1_min) / 200.
        X2_inc = (X2_max - X2_min) / 200.

        X1_surf = np.arange(X1_min, X1_max, X1_inc)
        X2_surf = np.arange(X2_min, X2_max, X2_inc)
        X1_surf, X2_surf = np.meshgrid(X1_surf, X2_surf)

        L_surf = model.predict(np.c_[X1_surf.ravel(), X2_surf.ravel()])
        L_surf = L_surf.reshape(X1_surf.shape)

        plt.title(classifier_name)
        plt.contourf(X1_surf, X2_surf, L_surf, cmap = plt.cm.coolwarm, zorder = 1)
        plt.scatter(X1, X2, s = 38, c = Y)

        plt.margins(0.0)
        # uncomment the following line to save images
        plt.savefig(f'{classifier_name}.png')
        plt.show()
        # plt.show(block=False)


if __name__ == "__main__":
    df = pd.read_csv('input.csv')
    models = Classifiers(df)
    print('Classifying with NN...')
    models.classifyNearestNeighbors()
    print('Classifying with Logistic Regression...')
    models.classifyLogisticRegression()
    print('Classifying with Decision Tree...')
    models.classifyDecisionTree()
    print('Classifying with Random Forest...')
    models.classifyRandomForest()
    print('Classifying with AdaBoost...')
    models.classifyAdaBoost()

    with open("output.csv", "w") as f:
        print('Name, Best Training Score, Testing Score',file=f)
        for line in models.outputs:
            print(line, file=f)