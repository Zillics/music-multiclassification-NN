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



def test():
	# Import data
	iris = datasets.load_iris()
	digits = datasets.load_digits()
	# Define model
	clf = svm.SVC(gamma=0.001, C=100.)
	# Train model
	clf.fit(digits.data[:-1], digits.target[:-1])
	# Test model
	pred = clf.predict(digits.data[-1:])
	print(pred)
	# Save model
	joblib.dump(clf, 'svm_models/test.joblib') 

def test_multi():
	iris = datasets.load_iris()
	X = iris.data[:, :2] # we only take the first two features.
	y = iris.target
	print(y)

	# Plot resulting Support Vector boundaries with original data
	# Create fake input data for prediction that we will use for plotting
	# create a mesh to plot in
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	h = (x_max / x_min)/100
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
	 np.arange(y_min, y_max, h))
	X_plot = np.c_[xx.ravel(), yy.ravel()]
	print(X_plot)
'''
	# Create the SVC model object
	C = 1.0 # SVM regularization parameter
	svc = svm.SVC(kernel='linear', C=C, decision_function_shape='ovr').fit(X, y)
	Z = svc.predict(X_plot)
	Z = Z.reshape(xx.shape)

	plt.figure(figsize=(15, 5))
	plt.subplot(121)
	plt.contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.3)
	plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1)
	plt.xlabel('Sepal length')
	plt.ylabel('Sepal width')
	plt.xlim(xx.min(), xx.max())
	plt.title('SVC with linear kernel')

	# Create the SVC model object
	C = 1.0 # SVM regularization parameter
	svc = svm.SVC(kernel='rbf', C=C, decision_function_shape='ovr').fit(X, y)

	Z = svc.predict(X_plot)
	Z = Z.reshape(xx.shape)

	plt.subplot(122)
	plt.contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.3)
	plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1)
	plt.xlabel('Sepal length')
	plt.ylabel('Sepal width')
	plt.xlim(xx.min(), xx.max())
	plt.title('SVC with RBF kernel')

	plt.show()
'''
if __name__ == '__main__':
	main()
