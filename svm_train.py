import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import dim_reduction, _to_categorical, normalize, soft_normalize
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.externals import joblib
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import cross_val_score

# TODO: POC version of Support Vector Machine
# TODO: POC version of kernel SVM 
# TODO: SVM + Grid search
# C = [2^1,2²,2³,2⁴...], gamma = [2⁻8,2⁻7.....]
# TODO: SVM + Bayesian Optimization

# Tweakable parameters
VAR_THR = 0.01
COR_THR = 0.94

def preprocessing(X_raw,y_raw):
	X = dim_reduction(X_raw,VAR_THR,COR_THR,print_status=True)
	X = soft_normalize(X)
	y = np.resize(y_raw,y_raw.shape[0])
	return X,y

def main():
	iris = datasets.load_iris()
	digits = datasets.load_digits()
	# Import data
	data = pd.read_csv('kaggle_data/train_data.csv',header=None)
	labels =  pd.read_csv('kaggle_data/train_labels.csv',header=None)
	# Preprocessing data
	X,y = preprocessing(data.values,labels.values)
	# Train-test split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=213)
	C_exps = [-2,-1,0,1,2,3,4,5,6,7]
	gamma_exps = [-7,-6,-5,-4,-3,-2]
	C = [2**exp for exp in C_exps]
	gamma = [2**exp for exp in gamma_exps]
	i_C = 0
	i_g = 0
	for C_i in C:
		i_g = 0
		for gamma_i in gamma:
			# shrinking? probability? tol?
			clf = SVC(C=C_i, kernel='rbf', degree=3, gamma=gamma_i, shrinking=True, probability=False, tol=0.001, cache_size=200, verbose=False)
			scores = cross_val_score(clf, X_train, X_test, cv=5)
			print("For C=2^%d, gamma=2^%d ; Accuracy: %0.2f (+/- %0.2f)" % (C_exps[i_C],gamma_exps[i_g],scores.mean(), scores.std() * 2))
			i_g +=1
		i_C += 1
	# Create model
	#clf = SVC(gamma='auto',verbose=True)
	#print(clf)
	# Train model
	#clf.fit(X_train, y_train)
	# Save model
	#joblib.dump(clf, 'svm_models/test.joblib')

if __name__ == '__main__':
	main()
