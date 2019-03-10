
########################################################## READ ME ##################################################################
#                                                                                                                                   #
#   This script was mainly used to test the keras model with the hyperopt / hyperas libs. This script doesn't have a purpose on     #
#   itself. I use the fmri data nonetheless, and it allowed me to conclude that Conv1D layers are useless regarding brain activity  #
#                                                       predictions                                                                 #
#                                                                                                                                   #
#####################################################################################################################################

################### Imports ###################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

from keras.models import Sequential
from keras.layers import Dense,Conv1D,Conv2D,MaxPooling2D,Flatten,MaxPooling1D,Dropout
from keras import optimizers

from nilearn import image,plotting
from nilearn.input_data import NiftiMasker
from nilearn.image import threshold_img
from nilearn.image import index_img
from nilearn.image import mean_img

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score,silhouette_score
from sklearn.cluster import KMeans

from hyperas import optim
from hyperas.distributions import choice, uniform

from hyperopt import Trials, STATUS_OK, tpe


data_path = '/home/brain/matthieu/relevant_data'


# utility functions 

def retrieve_data():

    #The hyperas / hyperopt libs needs two separate functions in order to work
    #One for building model and the other to define the data 
    data_path = '/home/brain/matthieu/relevant_data'
    path = os.path.join(data_path,'vector_nilearn.npz')
    data = np.load(path)
    X_train = data['a']
    X_test = data['b']
    Y_train = data['c']
    Y_test = data['d']

    #building the vectors 
    #we have to do it now otherwise the program can't recognize the vector in the building model function 
    #it raises UnboundLocalError: local variable 'X_train' referenced before assignment 
    X_train = np.expand_dims(X_train, axis=2)  
    X_test = np.expand_dims(X_test, axis=2) 

    return X_train,X_test,Y_train,Y_test

def model_building(X_train,X_test,Y_train,Y_test):

    #this function is used only for the optimization, it build only the model
    # fix random seed for reproducibility
    np.random.seed(7)
 
    X_dim = X_train.shape     
    Y_dim = Y_train.shape 

    print('Testing new model ...')

    # create model
    inp = (X_dim[1],1)
    model = Sequential()
    model.add(Conv1D(32, kernel_size=5, strides=1, activation={{choice(['relu', 'sigmoid','hard_sigmoid','softmax','tanh'])}}, input_shape=inp))
    model.add(MaxPooling1D(pool_size=2, strides=1))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Conv1D(64, kernel_size=5, activation={{choice(['relu', 'sigmoid','hard_sigmoid','softmax','tanh'])}}))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Conv1D(128, kernel_size=5, activation={{choice(['relu', 'sigmoid','hard_sigmoid','softmax','tanh'])}}))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Flatten())
    model.add(Dense({{choice([100, 250, 500, 1000])}}))
    model.add(Dense({{choice([100, 250, 500, 1000])}}))
    model.add(Dense({{choice([100, 250, 500, 1000])}}))
    model.add(Dense(len(Y_train.T), activation='linear'))
    
    #choose optimizer
    opt = optimizers.Adam(lr={{uniform(0,0.001)}}, beta_1={{uniform(0.8,1)}}, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    # Compile model
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
    #print(model.summary())
    
    # Fit / evaluate 
    model.fit(X_train, Y_train, epochs={{choice([50,75,100])}}, batch_size={{choice([25,50,75])}}, validation_split=0.2)
    
    score, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


def final_model(X_train,X_test,Y_train,Y_test,best_run):

    #build the model according to the best run done previously 

    print('\nBuilding the optimized model ... \n')
    #fix random seed for reproducibility
    np.random.seed(7)
 
    X_dim = X_train.shape     
    Y_dim = Y_train.shape 

    print('X vector shape: ', X_dim)
    print('Y vector shape: ', Y_dim)

    #the best run variable only gives us the index of the choice list input above in the model_building function 
    #so we have to re-create the choice list for all parameters 
    choice_activation = ['relu', 'sigmoid','hard_sigmoid','softmax','tanh']
    choice_dense = [100, 250, 500, 1000]
    choice_batch = [25,50,75]
    choice_epoch = [50,75,100]

    building_final_list_arg = ['activation','Dense','Dropout']
    #once the choices lists created we build the final choice list
    #i.e the lists which contains the actual choices of best run 
    epochs = choice_epoch[best_run['epochs']]
    batch_size = choice_batch[best_run['batch_size']]
    lr = best_run['lr']
    beta = best_run['beta_1']
    choice_activation_final = []
    choice_dense_final = []
    choice_dropout_final = []

    for key in best_run:
        #we check the type of the current element (whether its a dropout value, an activation function value ...) 
        #best_run dict is organized in such a way that the first key of a given element (dropout / activation function / dense size)
        #will be the first layer in which it appears (it is noted in this way: Dense (for first dense layer) ==> Dense_1 for the second ...)
        #so we just have to append the value at the end of each list
        if building_final_list_arg[0] in key:
            index = best_run[key]
            value = choice_activation[index]
            choice_activation_final.append(value)
        elif building_final_list_arg[1] in key:
            index = best_run[key]
            value = choice_dense[index]
            choice_dense_final.append(value)
        elif building_final_list_arg[2] in key:
            choice_dropout_final.append(best_run[key])
        else:
            pass

    #print(choice_dense_final, choice_dropout_final, choice_activation_final)

    # create model
    inp = (X_dim[1],1)
    model = Sequential()
    model.add(Conv1D(32, kernel_size=5, strides=1, activation=choice_activation_final[0], input_shape=inp))
    model.add(MaxPooling1D(pool_size=2, strides=1))
    model.add(Dropout(choice_dropout_final[0]))
    model.add(Conv1D(64, kernel_size=5, activation=choice_activation_final[1]))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(choice_dropout_final[1]))
    model.add(Conv1D(128, kernel_size=5, activation=choice_activation_final[2]))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(choice_dropout_final[2]))
    model.add(Flatten())
    model.add(Dense(choice_dense_final[0]))
    model.add(Dense(choice_dense_final[1]))
    model.add(Dense(choice_dense_final[2]))
    model.add(Dense(len(Y_train.T), activation='linear'))
    
    #choose optimizer
    opt = optimizers.Adam(lr=lr, beta_1=beta, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    # Compile model
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
    print(model.summary())
    
    # Fit / predict 
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    predictions = model.predict(X_test,verbose=1)
    scores = r2_score(Y_test, predictions, multioutput='raw_values')
    scores[scores < 0] = 0
    
    return scores,Y_dim

def plot_results(scores,data_path,Y_dim):

    #plot the result provided by the optimized model
    unique, counts = np.unique(scores, return_counts=True)
    maxi = max(unique)
    occurence = max(counts)
    print('\nscores max: ', maxi)
    print('scores equals to 0: ', occurence)
    print('scores above 0: ',Y_dim[1]-occurence,'\n')

    #plotting results 
    filename_mask = "/home/brain/datasets/SherlockMerlin_ds001110/sub-12/func/sub-12_task-SherlockMovie_bold_space-MNI152NLin2009cAsym_brainmask.nii.gz"
    masker = NiftiMasker(mask_img=filename_mask, detrend=True,standardize=True)
    masker.fit()
    meanepi = os.path.join(data_path,'meanepi.nii.gz')
    score_img = masker.inverse_transform(scores)
    plotting.plot_roi(score_img, bg_img=meanepi, title="Results of the clustering", cut_coords = 5, display_mode='z', aspect=1.25)
    plt.show()
    plt.close()


if __name__ == '__main__':

    best_run, best_model = optim.minimize(model=model_building, data=retrieve_data, algo=tpe.suggest, max_evals=5, trials=Trials())
    X_train,X_test,Y_train,Y_test = retrieve_data()
    #scores = model_building(X_train,X_test,Y_train,Y_test)
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("\nBest performing model chosen hyper-parameters:")
    print(best_run)

    scores,Y_dim = final_model(X_train,X_test,Y_train,Y_test,best_run)
    plot_results(scores,data_path,Y_dim)

   