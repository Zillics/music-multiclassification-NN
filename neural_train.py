import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from time import sleep



# Returns matrix X with features below/above variance and correlation thresholds reduced
def dim_reduction(X,variance_threshold,correlation_threshold,print_status=False):    
    # Check for nans
    if(print_status):
        print("Number of nans in data: ",np.isnan(X).sum())
    # Remove features with variance 0
    X_var = np.var(X,axis=0)
    remove_indices = np.where(X_var == 0)[0]
    if(print_status):
        print("Found ", len(remove_indices) ," features with variance 0: ", remove_indices, "....")

    # Normalize according to: x_norm = (x - x_min)/(x_max - x_min)
    X = np.delete(X,remove_indices,axis=1) # Delete in order to avoid division by 0 in normalization
    X_norm = (X - X.min(axis=0))/(X.max(axis=0) - X.min(axis=0))
    i = remove_indices[0]
    X_norm = np.insert(X_norm,[i,i,i,i],0,axis=1) # Add removed columns back in order to keep track of indices
    # Normalized variance
    X_norm_var = np.var(X_norm,axis=0)

    # Filter out features with small variance
    var_indices = np.where(X_norm_var <= variance_threshold)[0]
    if(print_status):
        print("Found ", len(var_indices), "features with normaliced variance lower than ", variance_threshold, ": ", var_indices)

    # Analyze correlations between features
    corr = np.corrcoef(X_norm.transpose())
    # For counting reasons: assign diagonal to arbitrary value
    for i in range(corr.shape[0]):
        corr[i,i] = -10
    # Check where there are strong correlations
    corr_indices = np.unique(np.where(corr > correlation_threshold)[0])
    if(print_status):
        print("Found ", len(corr_indices) ,"features with correlation stronger than ", correlation_threshold, " : \n",  corr_indices)
    remove_indices = np.unique(np.concatenate((corr_indices,var_indices)))
    if(print_status):
        print("Removing ", remove_indices.shape[0], " indices ")
    X = np.delete(X,remove_indices,axis=1)
    if(print_status):
    	print("Reduced shape: " , X.shape)
    return X

def _to_categorical(y_train):
	# One-hot encode labels
	y_train = np.resize(y_train,y_train.shape[0])
	encoder = LabelEncoder()
	encoder.fit(y_train)
	encoded_Y = encoder.transform(y_train)
	encoded_Y = np_utils.to_categorical(encoded_Y)
	return encoded_Y

# TODO: Figure out why no progress during training

# define baseline model
def create_model(n_features,regularizer):
	# create model
	model = Sequential()
	# Hidden layer 1
	print("input_dim: ", n_features)
	model.add(Dense(units=100, input_dim=n_features, activation='sigmoid',kernel_initializer='random_uniform',bias_initializer='zeros',kernel_regularizer=regularizers.l2(regularizer)))
	#model.add(Dense(units=100, activation='sigmoid'))
	model.add(Dense(units=64, activation='sigmoid'))
	# Output layer
	model.add(Dense(10, activation='softmax'))
	return model

def evaluate_model(model,X_train,y_train,X_test,y_test,epochs=10,batch_size=128,filename='model'):
	# Print basic model info
	model.summary()
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	# Train model
	print("INPUT DATA \n", X_train)
	print(X_train.shape)
	print("LABELS \n", y_train)
	print(y_train.shape)
	sleep(5)
	filepath=filename+"-{epoch:02d}-{loss:.4f}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min',period=50)
	callbacks_list = [checkpoint]
	history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=.1, verbose=True,callbacks=callbacks_list)
	# Evaluate model
	loss, accuracy  = model.evaluate(X_test, y_test, verbose=True)
	# Plot results
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['training', 'validation'], loc='best')
	plt.show()

	print()
	print(f'Test loss: {loss:.3}')
	print(f'Test accuracy: {accuracy:.3}')

def normalize(X):
	X_norm = (X - X.min(axis=0))/(X.max(axis=0) - X.min(axis=0))
	X_norm = X_norm*2 - 1
	return X_norm

# TODO: Batch normalization?
# TODO: Weight initialization
# TODO: Find optimal learning rate
# TODO: Tweak regularization parameters

def main():

	VAR_THR = 0.000
	COR_THR = 0.995
	REGUL = 0.005
	# Import data
	data = pd.read_csv('kaggle_data/train_data.csv',header=None)
	labels = pd.read_csv('kaggle_data/train_labels.csv',header=None)
	# Preprocessing
	X = dim_reduction(data.values,VAR_THR,COR_THR,print_status=True)
	X_norm = normalize(X)
	y = labels.values
	sleep(1)
	# Train-Test split
	X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=213)
	Y_train = _to_categorical(y_train)
	Y_test = _to_categorical(y_test)
	model = create_model(X_train.shape[1],REGUL)
	evaluate_model(model,X_train,Y_train,X_test,Y_test,epochs=1000,batch_size=32,filename="models/model_2/model_2_regul_005")
	preds = model.predict(X_test)
	print(preds)
	print(Y_test)
if __name__ == '__main__':
	main()