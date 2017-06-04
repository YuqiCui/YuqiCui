from __future__ import absolute_import
from __future__ import print_function
import h5py
import numpy as np
import scipy as sp
from sklearn import metrics

from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Dense, SpatialDropout2D, Activation, Flatten, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers.advanced_activations import ELU
from keras.regularizers import l1l2

def get_cnn_2d_1():
    np.random.seed(1007)
    model=Sequential()
    model.add(SpatialDropout2D(0.25,input_shape=(1,shapex,shapey)))
    model.add(Convolution2D(1, shapex, 5,input_shape=(1,shapex,shapey)))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.summary()
    return model
# input EEG dimensions
shapex, shapey = 16，59

# number of convolutional filters to use
nb_filters1 = 16
nb_filters2 = 4

# size of pooling area for max pooling
nb_pool = 2, 4

# other parameters
dropoutRate = .25
regRate = .01
seed = 1007

filename  = "driving_10s_40Hz_15_3S.mat"
savename  = "driving_10s_40Hz_15_3S_results.mat"

# change the path with whatever folder you are using
checkpointPath = '/driving.hdf5'
savepointPath  = '/driving_finalStep.hdf5'

# extract EEGs, RTs and nEvents from .mat
f = h5py.File(filename)
EEGs = np.transpose(np.array(f.get('EEGs')),(0,2,1)) # nEpochs * nChannels * nSamples
RTs = np.transpose(np.array(f.get('RTs')))
nEvents = np.array(f.get('nEvents'))

# Test each subject
MSE = np.zeros(len(nEvents))
for idxTest in range(len(nEvents)):
    if idxTest == 0:
        idxTestStart = 0
    else:
        idxTestStart = np.int(np.sum(nEvents[:idxTest-1]))
    idxTestEnd = idxTestStart + np.int(nEvents[idxTest])
    idsTrain = range(np.int(np.sum(nEvents)))
    idsTrain = np.delete(idsTrain,np.r_[idxTestStart:idxTestEnd],0)

    X_train = EEGs[idsTrain,:,:]
    X_test = EEGs[idxTestStart:idxTestEnd,:,:]
    # convert to float32 for Theano
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")

    X_train = X_train.reshape(X_train.shape[0], 1, shapex, shapey)
    X_test = X_test.reshape(X_test.shape[0], 1, shapex, shapey)

    Y_train = RTs[idsTrain]
    Y_test = RTs[idxTestStart:idxTestEnd]
    Y_train = Y_train.astype("int")
    Y_test = Y_test.astype("int")

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # set the random seed for reproducibility
    np.random.seed(seed)

    # start the model
    model = Sequential()

    ### start off the CNN.
    model.add(Convolution2D(nb_filters1, shapex, 1,
                            input_shape=(1, shapex, shapey),
                            W_regularizer = l1l2(l1=regRate, l2=regRate)))
    model.add(BatchNormalization(axis=1))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(SpatialDropout2D(dropoutRate))

    # reshape back to 2-d of size (Filters x Time)    
    permute_dims= 2, 1, 3
    model.add(Permute(permute_dims))

    ### second layer of CNN
    model.add(Convolution2D(nb_filters2, 2, 32, border_mode='same'))
    model.add(BatchNormalization(axis=1))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(nb_pool[0], nb_pool[1])))
    model.add(SpatialDropout2D(dropoutRate))

    ### third layer of CNN
    model.add(Convolution2D(nb_filters2, 8, 4,
                            border_mode='same'))
    model.add(BatchNormalization(axis=1))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(nb_pool[0], nb_pool[1])))
    model.add(SpatialDropout2D(dropoutRate))

    # reshape the output to a vector, then pass to sigmoid
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('linear'))
    
    model.compile(loss='mean_squared_error', optimizer="rmsprop", metrics=["mean_squared_error"])
    numParams = model.count_params()    

    # get the summary of the model
    model.summary()

    checkpointer = ModelCheckpoint(filepath=checkpointPath, verbose=1,
        save_best_only=True)

    # set up a model checkpoint to save the best model
    hist = model.fit(X_train, Y_train, batch_size=1000, nb_epoch=3,
        shuffle=True,verbose=1,validation_split=0.2,
        callbacks=[checkpointer])
    model.load_weights(checkpointPath)

    # get the probabilities for all trials in train/test/validation sets for later analysis
    output=model.predict(X_test,verbose=1,batch_size=1000)
    mse=Y_test-output
    MSE[idxTest]=np.sqrt(np.dot(mse.transpose(),mse)/len(Y_test))
    print(MSE)

## set up the callback for the model checkpointer
#checkpointer = ModelCheckpoint(filepath=checkpointPath, verbose=1, save_best_only=True)
#
## fit the model
#fittedModel = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=50, 
#                   verbose=0, callbacks=[checkpointer])
#
## save the weights at the last layer
#model.save_weights(savepointPath, overwrite=True)
#
## now load the optimal weights
#model.load_weights(checkpointPath)
#
## evaluate the model
#score = model.evaluate(X_test, Y_test, verbose=0)
#
#
## get the probabilities for all trials in train/test/validation sets for later analysis
#probs = model.predict_proba(X_test)
#probs_train = model.predict_proba(X_train)
#predicted = model.predict_classes(X_test)
#print('Test score:', score[0])
#print('Test accuracy:', score[1])
#
## calculate AUC
#fpr, tpr, thresholds = metrics.roc_curve(Y_test[:,1], probs[:,1], pos_label=1)
#AUC = metrics.auc(fpr, tpr)
#print(AUC)
#
## get the learned spatial filters
#W = model.layers[0].W.get_value(borrow=True)
#W = np.squeeze(W)
#
#### store relevant variables into MATLAB file
#results = fittedModel.history
#results['W']               = W
#results['numParams']       = numParams
#results['dropoutRate']     = dropoutRate
#results['batch_size1']     = batch_size
#results['AUC']             = AUC
#results['prob_test']       = probs
#results['prob_train']      = probs_train
#results['probs_val']       = probs_val
#results['reg_rate']        = regRate
#results['score']           = score
#results['score_val']       = score_val
#
## save the file into a MATLAB structure
#sp.io.savemat(savename, results)
