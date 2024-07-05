
import csv
import nltk
from nltk.tokenize import LineTokenizer
import numpy as np
import math
import os
import string
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pandas import Series
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import statistics

def train_test_indexes(fold_indexes, current_test_index):
    train_indexes = []
    test_indexes = fold_indexes[current_test_index]

    for i in range(0, len(fold_indexes)):
        if i != current_test_index:
            train_indexes += fold_indexes[i]

    return train_indexes, test_indexes

def create_fold_indexes(df, n_folds=10):
    random_indexes = df.sample(frac=1).index

    chunk_size = len(random_indexes) // n_folds  # maybe an issue
    parent_list = []
    child_list = []

    for a in range(n_folds):
        # go through one more time
        for i in range(chunk_size):
            child_list.append(random_indexes[(a * chunk_size) + i])

        parent_list.append(child_list)
        child_list = []

    remainder = len(random_indexes) % n_folds 
    if remainder != 0:
        for i in range(1, remainder + 1):
            parent_list[i - 1].append(random_indexes[-i])

    return parent_list

def calc_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def main():
    data = pd.read_csv('./data/output.csv', index_col=0)

    #Create X and Y values
    Y = data.loc[:,'Popularity']
    X = data.drop(columns=['Popularity', 'Songs', "Genre", "Streams"])
    X = sm.add_constant(X)

    print(X)
    print(Y)
    # Create model
    # model1 = sm.OLS(Y, X.astype(float)).fit()
    # print(model1.summary2())

    # KNN
    np.random.seed(42)
    boxplot_data = []
    k_bucket = []

    # Confusion Matrix
    prediction_matrix = pd.DataFrame([[0,0,0,0] for i in range(len(Y.unique()))])
    indexes = Y.unique()
    indexes.sort()
    prediction_matrix.columns = indexes
    prediction_matrix.index = indexes

    fold_indexes = create_fold_indexes(data)
    print("Fold indexes", fold_indexes)
    for k in range(1, 11, 2):
        fold_accuracy = []
        for i in range(0, 10):
            train_indexes, test_indexes = train_test_indexes(fold_indexes, i)
            print(train_indexes)
            print(test_indexes)
            X_train = X.iloc[train_indexes]
            Y_train = Y.iloc[train_indexes]  
            X_test =  X.iloc[test_indexes] 
            Y_test =  Y.iloc[test_indexes]
            '''
            model = sm.OLS(Y_train, X_train).fit()
            # print(model.summary2())
            Y_pred = model.predict(X_test)
            rmse = calc_rmse(Y_test, Y_pred)
            fold_accuracy.append(rmse)

            plt.scatter(Y_pred, Y_test)
            plt.xlabel("Predicted Streams")
            plt.ylabel("Actual Streams")
            plt.show()
            '''
            #standardize
            scaler = StandardScaler()
            scaler.fit(X_train) 
            X_train = scaler.transform(X_train)

            # knn
            knn = KNeighborsClassifier(n_neighbors= k)
            knn.fit(X_train, Y_train)

            # standard test
            X_test = scaler.transform(X_test)
            Y_pred = knn.predict(X_test)

            acc = accuracy_score(Y_test, Y_pred)
            fold_accuracy.append(acc)
            for j in range(len(Y_test)):
                prediction_matrix.loc[Y_test.iloc[j]][Y_pred[j]] = prediction_matrix.loc[Y_test.iloc[j]][Y_pred[j]] + 1
        k_bucket.append(k)
        boxplot_data.append(fold_accuracy)
        print("k=", k, "mean accuracy=", statistics.mean(fold_accuracy), "std=", statistics.stdev(fold_accuracy))
    boxplot_df = pd.DataFrame(boxplot_data)
    boxplot_df["index"] = range(0, 5)
    boxplot_df.set_index("index", inplace=True)
    boxplot_df.T.plot.box()
    plt.title("Choice of K vs. Prediction Accuracy")
    plt.ylabel("prediction accuracy")
    plt.xlabel("k neighbors")
    plt.show()

    print("Confusion Matrix\n", prediction_matrix)

main()