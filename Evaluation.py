import pandas as pd
from sklearn import preprocessing, model_selection, metrics
import numpy as np
from joblib import load


def evaluate(classifier, test, label):
    print("CLASSIFICATION RESULTS:\n")
    print("Accuracy sul test: ",metrics.accuracy_score(label, classifier.predict(test)))
    print()
    print("Precision sul test:", metrics.precision_score(label, classifier.predict(test), average="macro"))
    print()
    print("Recall sul test: ", metrics.recall_score(label, classifier.predict(test), average="macro"))
    print()
    print("F1 sul test:", metrics.f1_score(label, classifier.predict(test), average="macro"))
    print()
    print("Matrice di confusione: \n", metrics.confusion_matrix(label, classifier.predict(test)))


def scaling(train, test):
    scaler = preprocessing.MinMaxScaler()
    #scaler = preprocessing.RobustScaler()
    scaler.fit(
        train)
    test = scaler.transform(test)
    return test


def featureSelection(test):
    mask = np.array(
        [False, True, True, True, True, True, True, True, False, False, True, True, True, True, True, True, False,
         True, False, True]) #maschera ottenuta dalla fase di training. Pone a False gli indici delle feature da scartare
    return test[:, mask]


def get_na_count(dataset):
    boolean_mask = dataset.isna()
    return boolean_mask.sum(axis=0)


if __name__ == "__main__":
    trainingDataset = pd.read_csv('training_set.csv')
    testDataset = pd.read_csv('test.csv')  # inserire il nome del file di test

    print("Dati mancanti: ")
    print(get_na_count(testDataset))
    print("-------------------")
    for i in range(1, 21):
        mean_test = testDataset["F" + str(i)].mean()
        testDataset["F" + str(i)] = testDataset["F" + str(i)].fillna(mean_test)
        mean_train = trainingDataset["F" + str(i)].mean()
        trainingDataset["F" + str(i)] = trainingDataset["F" + str(i)].fillna(mean_train)
    print("Dati mancanti dopo il pre processamento: ")
    print(get_na_count(testDataset))
    print("-------------------")
    x = trainingDataset.iloc[:, 0:20].values #separazione dei dati dalle label
    y = trainingDataset.iloc[:, 20].values
    x_test = trainingDataset.iloc[:, 0:20].values
    y_test = trainingDataset.iloc[:, 20].values
    train_x = model_selection.train_test_split(x, y, test_size=0.2, random_state=0)[0] #ottengo il dataset con cui ho addestrato il modello, in modo da poter fare sca
    x_test = scaling(train_x, x_test)
    # feature selection
    x_test = featureSelection(x_test)
    print("Test shape dopo feature selection: ")
    print(x_test.shape)
    print("-------------------")
    clf = load('mlpClassifier_minMaxScaler.joblib')
    evaluate(clf, x_test, y_test)
