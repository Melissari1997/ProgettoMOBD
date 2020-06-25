import pandas as pd
import sklearn.metrics as metrics
import sklearn.model_selection as model_selection
import sklearn.preprocessing as preprocessing
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from joblib import dump
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.feature_selection import mutual_info_classif

from ParamsTuning import mlp_param_selection, svm_param_selection, random_forest_param_selection, knn_param_selection, \
    sgd_param_selection


def get_na_count(dataset):
    boolean_mask = dataset.isna()
    return boolean_mask.sum(axis=0)


dataset = pd.read_csv('training_set.csv')
dataset.describe(include='all')
summary_test = get_na_count(dataset)
print("Dati mancanti: ")
print(get_na_count(dataset))
print("-------------------")
for i in range(1, 21):
    mean = dataset["F" + str(i)].mean()
    dataset["F" + str(i)] = dataset["F" + str(i)].fillna(mean)
print("Dati mancanti dopo il pre processamento: ")
print(get_na_count(dataset))
print("-------------------")
counts = dataset['CLASS'].value_counts()
class1 = counts[0]
class2 = counts[1]
class3 = counts[2]
class4 = counts[3]
print('Frazione di classe 1', round(class1 / (class1 + class2 + class3 + class4), 4))
print('Frazione di classe 2', round(class2 / (class1 + class2 + class3 + class4), 4))
print('Frazione di classe 3', round(class3 / (class1 + class2 + class3 + class4), 4))
print('Frazione di classe 4', round(class4 / (class1 + class2 + class3 + class4), 4))
print("-------------------")

#separo i dati dalle label
x = dataset.iloc[:, 0:20].values
y = dataset.iloc[:, 20].values
# divido il dataset in training e test
train_x, test_x, train_y, test_y = model_selection.train_test_split(x, y, test_size=0.2, random_state=0)

#faccio lo scaling dei dati
scaler = preprocessing.MinMaxScaler()
scaler.fit(train_x)  # utilizzo la media e la varianza del training set per non avere overfitting
train_x = scaler.transform(train_x)
test_x = scaler.transform(test_x)


#elimino le 5 features che hanno minor dipendenza con la feature di target (cioè la classe di appartenenza)
sel = SelectKBest(mutual_info_classif, k=15)
sel.fit(train_x, train_y)
train_x = sel.transform(train_x)
print("Feature index after SelectKBest: ", sel.get_support(indices=True))
test_x = sel.transform(test_x)
print('Train shape after Feature Selection:', train_x.shape, train_y.shape)
print('Test shape after Feature Selection::', test_x.shape, test_y.shape)
print("-------------------")
#oversampling con SMOTE
sm = SMOTE()
#sm = RandomUnderSampler()
#sm = RandomOverSampler()
#sm = ADASYN(sampling_strategy="not majority")
train_x , train_y = sm.fit_sample(train_x, train_y)


print('Train shape after balancing:', train_x.shape, train_y.shape)
print('Test shape after balancing:', test_x.shape, test_y.shape)
print("-------------------")


mlpClassifier = mlp_param_selection(train_x, train_y, n_folds=10, metric='f1_macro') #tuning degli iperparametri
print("F1 for MLP:", metrics.f1_score(test_y, mlpClassifier.predict(test_x), average="macro"))
dump(mlpClassifier,'mlpClassifier.joblib')
print("-------------------")
svm_classifier = svm_param_selection(train_x, train_y, n_folds=10, metric='f1_macro')
print("F1 for SVM :", metrics.f1_score(test_y, svm_classifier.predict(test_x), average="macro"))
dump(svm_classifier,'svmClassifier.joblib')
print("-------------------")
randomF_classifier = random_forest_param_selection(train_x, train_y, n_folds=10, metric='f1_macro')
print("F1 for Random Forest", metrics.f1_score(test_y, randomF_classifier.predict(test_x), average="macro"))#dump(randomF_classifier,'randomF_classifier2.joblib')
dump(randomF_classifier,'randomF_classifier.joblib')
print("-------------------")
knn_classifier = knn_param_selection(train_x, train_y, n_folds=10, metric='f1_macro')
print("F1 for KNN", metrics.f1_score(test_y, randomF_classifier.predict(test_x), average="macro"))
dump(randomF_classifier,'knnClassifier.joblib')
print("-------------------")
sgdClassifier = sgd_param_selection(train_x, train_y, n_folds=10, metric='f1_macro')
print("F1 for SGD:", metrics.f1_score(test_y, sgdClassifier.predict(test_x), average="macro"))
dump(sgdClassifier,'sgdClassifier.joblib')
