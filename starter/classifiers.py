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
        plt.legend(['A', 'B'])
        plt.savefig('partA.png')
        plt.show()

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
        pass
        # Use the following line to plot the results
        # self.plot(self.testing_data, clf.predict(self.testing_data),model=clf,classifier_name=name)

    def classifyNearestNeighbors(self):
        # TODO: Write code to run a Nearest Neighbors classifier
        avgScore = None
        bestParams = None
        for n_neighbors in range(1, 20, 2):
            for leaf_size in range(5, 31, 5):
                total_score = 0
                for i in range(5):
                    trainIdx = [j for j in range(5) if j != i]
                    validationIdx = i

                    trainData = np.vstack([self.foldDatas[j] for j in trainIdx])
                    trainLabels = np.hstack([self.foldLabels[j] for j in trainIdx])

                    validationData = self.foldDatas[validationIdx]
                    validationLabels = self.foldLabels[validationIdx]

                    clf = KNeighborsClassifier(n_neighbors=n_neighbors, leaf_size=leaf_size)
                    clf.fit(trainData, trainLabels)
                    score = clf.score(validationData, validationLabels)
                    total_score += score

                newAvg = total_score / 5
                print(f'NN with n_neighbors={n_neighbors}, leaf_size={leaf_size} has average score: {newAvg}')
                if avgScore is None or newAvg > avgScore:
                    avgScore = newAvg
                    bestParams = (n_neighbors, leaf_size)

        print(f'Best NN params: n_neighbors={bestParams[0]}, leaf_size={bestParams[1]} with average score: {avgScore}')

        clf = KNeighborsClassifier(n_neighbors=bestParams[0], leaf_size=bestParams[1])
        clf.fit(self.training_data, self.training_labels)
        trainingScore = clf.score(self.training_data, self.training_labels)
        print(f'Final NN Training Score: {trainingScore}')
        testingScore = clf.score(self.testing_data, self.testing_labels)
        print(f'Final NN Testing Score: {testingScore}')

        self.outputs.append(f'K-Nearest Neighbors, {trainingScore}, {testingScore}')

        self.plot(self.testing_data, self.testing_labels, model=clf, classifier_name='K-Nearest Neighbors')

    def classifyLogisticRegression(self):
        # TODO: Write code to run a Logistic Regression classifier
        avgScore = None
        bestParams = None
        for c in [0.1, 0.5, 1, 5 ,10, 50, 100]:
            total_score = 0
            for i in range(5):
                trainIdx = [j for j in range(5) if j != i]
                validationIdx = i

                trainData = np.vstack([self.foldDatas[j] for j in trainIdx])
                trainLabels = np.hstack([self.foldLabels[j] for j in trainIdx])

                validationData = self.foldDatas[validationIdx]
                validationLabels = self.foldLabels[validationIdx]

                clf = LogisticRegression(C=c)
                clf.fit(trainData, trainLabels)
                score = clf.score(validationData, validationLabels)
                total_score += score

            newAvg = total_score / 5
            print(f'Logistic Regression with C={c} has average score: {newAvg}')
            if avgScore is None or newAvg > avgScore:
                avgScore = newAvg
                bestParams = c

        print(f'Best Logistic Regression params: c={bestParams[0]} with average score: {avgScore}')

        clf = LogisticRegression(C=bestParams)
        clf.fit(self.training_data, self.training_labels)
        trainingScore = clf.score(self.training_data, self.training_labels)
        print(f'Final Logistic Regression Training Score: {trainingScore}')
        testingScore = clf.score(self.testing_data, self.testing_labels)
        print(f'Final Logistic Regression Testing Score: {testingScore}')

        self.outputs.append(f'Logistic Regression, {trainingScore}, {testingScore}')

        self.plot(self.testing_data, self.testing_labels, model=clf, classifier_name='Logistic Regression')

    def classifyDecisionTree(self):
        # TODO: Write code to run a Logistic Regression classifier
        avgScore = None
        bestParams = None
        for max_depth in range(1, 20, 2):
            for min_samples_split in range(5, 31, 5):
                total_score = 0
                for i in range(5):
                    trainIdx = [j for j in range(5) if j != i]
                    validationIdx = i

                    trainData = np.vstack([self.foldDatas[j] for j in trainIdx])
                    trainLabels = np.hstack([self.foldLabels[j] for j in trainIdx])

                    validationData = self.foldDatas[validationIdx]
                    validationLabels = self.foldLabels[validationIdx]

                    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
                    clf.fit(trainData, trainLabels)
                    score = clf.score(validationData, validationLabels)
                    total_score += score

                newAvg = total_score / 5
                print(f'Decision Tree with max_depth={max_depth}, min_samples_split={min_samples_split} has average score: {newAvg}')
                if avgScore is None or newAvg > avgScore:
                    avgScore = newAvg
                    bestParams = (max_depth, min_samples_split)

        print(f'Best Decision Tree params: n_neighbors={bestParams[0]}, leaf_size={bestParams[1]} with average score: {avgScore}')

        clf = DecisionTreeClassifier(max_depth=bestParams[0], min_samples_split=bestParams[1])
        clf.fit(self.training_data, self.training_labels)
        trainingScore = clf.score(self.training_data, self.training_labels)
        print(f'Final Decision Tree Training Score: {trainingScore}')
        testingScore = clf.score(self.testing_data, self.testing_labels)
        print(f'Final Decision Tree Testing Score: {testingScore}')

        self.outputs.append(f'Decision Tree, {trainingScore}, {testingScore}')

        self.plot(self.testing_data, self.testing_labels, model=clf, classifier_name='Decision Tree')

    def classifyRandomForest(self):
        # TODO: Write code to run a Random Forest classifier
        avgScore = None
        bestParams = None
        for max_depth in range(1, 20, 2):
            for min_samples_split in range(5, 31, 5):
                total_score = 0
                for i in range(5):
                    trainIdx = [j for j in range(5) if j != i]
                    validationIdx = i

                    trainData = np.vstack([self.foldDatas[j] for j in trainIdx])
                    trainLabels = np.hstack([self.foldLabels[j] for j in trainIdx])

                    validationData = self.foldDatas[validationIdx]
                    validationLabels = self.foldLabels[validationIdx]

                    clf = RandomForestClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
                    clf.fit(trainData, trainLabels)
                    score = clf.score(validationData, validationLabels)
                    total_score += score

                newAvg = total_score / 5
                print(f'Random Forest with max_depth={max_depth}, min_samples_split={min_samples_split} has average score: {newAvg}')
                if avgScore is None or newAvg > avgScore:
                    avgScore = newAvg
                    bestParams = (max_depth, min_samples_split)

        print(f'Best Random Forest params: n_neighbors={bestParams[0]}, leaf_size={bestParams[1]} with average score: {avgScore}')

        clf = RandomForestClassifier(max_depth=bestParams[0], min_samples_split=bestParams[1])
        clf.fit(self.training_data, self.training_labels)
        trainingScore = clf.score(self.training_data, self.training_labels)
        print(f'Final Random Forest Training Score: {trainingScore}')
        testingScore = clf.score(self.testing_data, self.testing_labels)
        print(f'Final Random Forest Testing Score: {testingScore}')

        self.outputs.append(f'Random Forest, {trainingScore}, {testingScore}')

        self.plot(self.testing_data, self.testing_labels, model=clf, classifier_name='Random Forest')

    def classifyAdaBoost(self):
        # TODO: Write code to run a AdaBoost classifier
        avgScore = None
        bestParams = None
        for number_of_estimators in [0.1, 0.5, 1, 5 ,10, 50, 100]:
            total_score = 0
            for i in range(5):
                trainIdx = [j for j in range(5) if j != i]
                validationIdx = i

                trainData = np.vstack([self.foldDatas[j] for j in trainIdx])
                trainLabels = np.hstack([self.foldLabels[j] for j in trainIdx])

                validationData = self.foldDatas[validationIdx]
                validationLabels = self.foldLabels[validationIdx]

                clf = AdaBoostClassifier(n_estimators=number_of_estimators)
                clf.fit(trainData, trainLabels)
                score = clf.score(validationData, validationLabels)
                total_score += score

            newAvg = total_score / 5
            print(f'Ada Boost with number_of_estimators={number_of_estimators} has average score: {newAvg}')
            if avgScore is None or newAvg > avgScore:
                avgScore = newAvg
                bestParams = number_of_estimators

        print(f'Best Ada Boost params: number_of_estimators={bestParams[0]} with average score: {avgScore}')

        clf = AdaBoostClassifier(n_estimators=bestParams)
        clf.fit(self.training_data, self.training_labels)
        trainingScore = clf.score(self.training_data, self.training_labels)
        print(f'Final Ada Boost Training Score: {trainingScore}')
        testingScore = clf.score(self.testing_data, self.testing_labels)
        print(f'Final Ada Boost Testing Score: {testingScore}')

        self.outputs.append(f'Ada Boost, {trainingScore}, {testingScore}')

        self.plot(self.testing_data, self.testing_labels, model=clf, classifier_name='Ada Boost')

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