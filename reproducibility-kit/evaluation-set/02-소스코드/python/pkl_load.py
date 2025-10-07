#!/usr/bin/env python3

# https://m.blog.naver.com/wideeyed/221330321950


# conda install -c anaconda scikit-learn
import gzip
import pickle
import joblib
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

#   Paths
dataSetPath = './Dataset/'
featureSetsPath = './Dataset/FeatureSet/'

file_name = 'X_test_noScale.pkl'
# file_name = 'zscore_scaler.pkl'
obj = joblib.load(featureSetsPath + file_name)

# Load data files
#with open(featureSetsPath + 'X_training_noScale.pkl', 'rb') as f:
#    vector_load = pickle.load(f)

#train_X = pickle.load(open(featureSetsPath + "X_training_noScale.pkl", "rb"))
#valid_X = pickle.load(open(featureSetsPath + "X_validation_noScale.pkl", "rb"))
#test_X = pickle.load(open(featureSetsPath + "X_test_noScale.pkl", "rb"))

#train_Y = pickle.load(open(featureSetsPath + "Y_training.pkl", "rb"))
#valid_Y = pickle.load(open(featureSetsPath + "Y_validation.pkl", "rb"))
#test_Y = pickle.load(open(featureSetsPath + "Y_test.pkl", "rb"))

train_X = joblib.load(featureSetsPath + "X_training_noScale.pkl")
valid_X = joblib.load(featureSetsPath + "X_validation_noScale.pkl")
test_X  = joblib.load(featureSetsPath + "X_test_noScale.pkl")

train_Y = joblib.load(featureSetsPath + "Y_training.pkl")
valid_Y = joblib.load(featureSetsPath + "Y_validation.pkl")
test_Y  = joblib.load(featureSetsPath + "Y_test.pkl")

# Load an object file (=scaler object) that is saved with a pickled binary file format
file_name = 'zscore_scaler.pkl'
scaler = joblib.load(featureSetsPath + file_name)

# print('Loaded scaler results:', scaler.fit_transform(train_X))
# print('Loaded scaler results:', scaler.transform(train_X))
# train_X_scaled = scaler.transform(train_X)
# train_X_scaled[:2]

# transform both datasets
X_train_scaled = scaler.transform(train_X)
X_test_scaled = scaler.transform(test_X)

# summarize the scale of each input variable
for i in range(test_X.shape[1]):
    print('>%d, train: min=%.3f, max=%.3f, test: min=%.3f, max=%.3f' %
          (i, X_train_scaled[:, i].min(), X_train_scaled[:, i].max(),
           X_test_scaled[:, i].min(), X_test_scaled[:, i].max()))
