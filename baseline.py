

########################################################## READ ME ##################################################################
#                                                                                                                                   #
#   Use the results of the k means clustering and try to replace the estimator (Ridge(alpha)) with either an MLP                    #
#   From sklearn or a deep neural network model from Keras. The model from keras can be optimized if you enter the right arguments  #
#                                                                                                                                   #
#####################################################################################################################################


################### Imports ###################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os,argparse

from nilearn import image,plotting
from nilearn.input_data import NiftiMasker
from nilearn.image import threshold_img
from nilearn.image import index_img
from nilearn.image import mean_img
from nilearn.input_data import NiftiLabelsMasker

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score,silhouette_score
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor

from tqdm import tqdm 

from check_mask import create_intersect 

from keras.models import Sequential
from keras.layers import Dense,Conv1D,Conv2D,MaxPooling2D,Flatten,MaxPooling1D,Dropout
from keras import optimizers

from hyperas import optim
from hyperas.distributions import choice, uniform

from hyperopt import Trials, STATUS_OK, tpe

################### Global variables ###################

layer = 20 
subject = 12
main_path = '/home/brain/matthieu/test_baseline'
data_path = '/home/brain/matthieu/relevant_data'

################### Utility functions ###################


def get_parser():

    #we code the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_model', help = 'Test with the baseline MLPRegressor, if not, test with the keras equivalent of this MLPRegressor', 
        action = 'store_true')
    parser.add_argument('-o', '--optimized', help = 'Optimize the keras network before usage, cannot be used with the -b argument', 
        action = 'store_true')

    return parser


def retrieve_data(loaded_stimuli,fmri_ready):

    middle = int(loaded_stimuli.shape[0]/2)
    print('Shape of fmri_ready: ', fmri_ready.shape)
    y_train = fmri_ready[:middle] 
    y_test = fmri_ready[middle:]
    X_train = (loaded_stimuli[:middle])
    X_test = (loaded_stimuli[middle:])

    return X_train,y_train,X_test,y_test


def model_optimization(X_train,y_train,X_test,y_test):

    X_dim = X_train.shape     
    Y_dim = y_train.shape 

    print('New X shape: ',X_dim)
    print('New Y shape: ',Y_dim,'\n')
    # fix random seed for reproducibility
    np.random.seed(7)

    print('Testing model ...')

    # create model
    model = Sequential()
    model.add(Dense(X_dim[1],input_shape=(X_dim[1],), activation={{choice(['relu', 'sigmoid','hard_sigmoid','softmax','tanh'])}}))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([100, 250, 500, 1000])}}, activation={{choice(['relu', 'sigmoid','hard_sigmoid','softmax','tanh'])}}))
    model.add(Dense(Y_dim[1], activation='relu'))
    
    # If we choose 'two', add an additional dense layer
    if {{choice(['one', 'two'])}} == 'two':
        model.add(Dense(250,activation='relu'))


    #define the optimizer
    opt = optimizers.Adam(lr={{uniform(0,0.001)}}, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)

    # Compile model
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

    #Fit / evaluate model
    model.fit(X_train, y_train, epochs={{choice([100,200,300])}}, batch_size={{choice([25,50,75])}}, validation_split=0.1)

    score, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test accuracy:', acc)

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


def optimized_model(best_run,X_train,y_train,X_test,y_test):

    #build the optimized model 
    #the best run variable only gives us the index of the choice list input above in the model_building function 
    #so we have to re-create the choice list for all parameters 

    choice_activation = ['relu', 'sigmoid','hard_sigmoid','softmax','tanh']
    choice_dense = [100, 250, 500, 1000]
    choice_batch = [25,50,75]
    choice_epoch = [100,200,300]

    building_final_list_arg = ['activation']

    #once the choices lists created we build the final choice list
    #i.e the lists which contains the actual choices of best run 
    epochs = choice_epoch[best_run['epochs']]
    batch_size = choice_batch[best_run['batch_size']]
    lr = best_run['lr']
    droprate = best_run['Dropout']
    number_dense_layer = best_run['add']
    dense_size = best_run['Dense']

    choice_activation_final = []
    choice_dense_final = []

    for key in best_run:
        #we check the type of the current element (whether its a dropout value, an activation function value ...) 
        #best_run dict is organized in such a way that the first key of a given element (dropout / activation function / dense size)
        #will be the first layer in which it appears (it is noted in this way: Dense (for first dense layer) ==> Dense_1 for the second ...)
        #so we just have to append the value at the end of each list
        if building_final_list_arg[0] in key:
            index = best_run[key]
            value = choice_activation[index]
            choice_activation_final.append(value)
        else:
            pass

    #we build the model 
    model = Sequential()

    model.add(Dense(X_dim[1],input_shape=(X_dim[1],), activation=choice_activation_final[0]))
    model.add(Dropout(droprate))
    model.add(Dense(dense_size, activation=choice_activation_final[1]))
    if add == 'two':
        model.add(Dense(250,activation='relu'))
    model.add(Dense(Y_dim[1], activation='relu'))

    #choose optimizer
    opt = optimizers.Adam(lr=lr, beta_1=beta, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    # Compile model
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
    print(model.summary())
    
    # Fit / predict 
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    predictions = model.predict(X_test,verbose=1)

    return predictions

def keras_equivalent(X_train,y_train,X_test,y_test):
 
    X_dim = X_train.shape     
    Y_dim = y_train.shape 

    print('New X shape: ',X_dim)
    print('New Y shape: ',Y_dim,'\n')
    # fix random seed for reproducibility
    np.random.seed(7)

    print('Testing model ...')

    # create model
    #inp = (X_dim[1],1)
    model = Sequential()
    model.add(Dense(X_dim[1],input_shape=(X_dim[1],), activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(Y_dim[1], activation='relu'))
    #define the optimizer
    opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)

    # Compile model
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
    print(model.summary())

    # Fit / predict 
    model.fit(X_train, y_train, epochs=200, batch_size=50, validation_split=0.1)
    
    predictions = model.predict(X_test,verbose=1)

    return predictions


def create_saving_folder(outputpath):

    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
        print("New folder created")



def init(subject,layer,filename_irm,filename_mask,filename_stimuli):

    #we transform the layer and subject variables in order to use them in the input path 
    if subject < 10:
        subject = '0' + str(subject)
    else:
        subject = str(subject)
    
    if layer < 10:
        layer = '0' + str(layer)
    else:
        layer = str(layer)

    print("Initializing tests for subject " + subject + " and layer " + layer + " ...")
    #we dig out our data from these files and we create the mask in order to have a better rendering in the future plots
    
    meanepi = (mean_img(filename_irm))
    loaded_stimuli = np.load(filename_stimuli)
    masker = NiftiMasker(mask_img=filename_mask, detrend=True,standardize=True)
    masker.fit()
    fmri_data = masker.transform(filename_irm)
    fmri_ready = fmri_data[17:-(fmri_data.shape[0]-17-loaded_stimuli.shape[0])]

    # building the encoding models
    middle = int(loaded_stimuli.shape[0]/2)
    y_train = fmri_ready[:middle] 
    y_test = fmri_ready[middle:]
    X_train = (loaded_stimuli[:middle])
    X_test = (loaded_stimuli[middle:])
    print("Init done")

    return X_train,X_test,y_train,y_test,masker,meanepi


def reference_model_kmeans(X_train,X_test,y_train,y_test,estimator):

    estimator.fit(X_train,y_train)
    predictions = estimator.predict(X_test)
    scores = r2_score(y_test, predictions, multioutput='raw_values')
    scores[scores < 0] = 0

    return scores


def find_cluster(label,masker,data_path,nbc):

    masked_label = np.zeros_like(label) 
    #nbc = 4
    masked_label[label == nbc] = 1
    print('\nData loaded successfully')
    filename_mask = os.path.join(data_path,'mask_inter.nii.gz')
    masker = NiftiMasker(mask_img=filename_mask, detrend=True,standardize=True)
    masker.fit()

    #now that we have the scores we create the masks
    score_map_img = masker.inverse_transform(masked_label)
    plotting.plot_roi(score_map_img, bg_img=meanepi, title="Results of the clustering", 
        cut_coords = 5, display_mode='z', aspect=1.25)
    plt.close()
    unique, counts = np.unique(masked_label, return_counts=True)
    print(dict(zip(unique,counts)))

    score_map_img = masker.inverse_transform(masked_label)
    masker = NiftiMasker(mask_img=score_map_img, detrend=True,standardize=True)
    masker.fit()

    return masker

def second_processing(masker,filename_irm,loaded_stimuli,main_path,meanepi,alpha,nbc,num_clust,best_run=None):
   
    loaded_stimuli = np.load(filename_stimuli)
    fmri_data = masker.transform(filename_irm)
    fmri_ready = fmri_data[17:-(fmri_data.shape[0]-17-loaded_stimuli.shape[0])]
    X_train,y_train,X_test,y_test = retrieve_data(loaded_stimuli,fmri_ready)

    #getting the scores 
    if base_model == True:
        if optimized == True: 
            print('Useless argument -o provided. Using the sklearn MLP nonetheless...') 
        else: 
            pass
        estimator = MLPRegressor(hidden_layer_sizes=(500),verbose=True,batch_size=50,alpha=0.1,learning_rate_init=0.0001)
        estimator.fit(X_train,y_train)
        predictions = estimator.predict(X_test)

    else: 
        
        if optimized == True: 
            predictions = optimized_model(best_run,X_train,y_train,X_test,y_test)
        #Use the basic keras model
        else:
            predictions = keras_equivalent(X_train,y_train,X_test,y_test) 
    
    #we process the scores 
    scores = r2_score(y_test, predictions, multioutput='raw_values')
    scores[scores < 0] = 0

    #we display the important results 
    Y_dim = y_train.shape
    unique, counts = np.unique(scores, return_counts=True)
    maxi = max(unique)
    occurence = max(counts)
    print('\nscores max: ', maxi)
    print('scores equals to 0: ',occurence)
    print('scores above 0: ',Y_dim[1]-occurence,'\n')
    plt.figure()

    #plotting the results
    mean_score_map_img = masker.inverse_transform(scores)
    mean_img = threshold_img(mean_score_map_img, threshold=1e-6)

    #Plot the mean image of all images obtained by the different segmentation  
    plotting.plot_stat_map(mean_img, bg_img=meanepi, cut_coords=5, display_mode='z', aspect=1.25, threshold=1e-6,
        title="Mean image")

    #We save the image 
    img2_name = 'mean_img_label_'+str(nbc)+'.png'
    img2_path = os.path.join(main_path,img2_name)
    plt.savefig(img2_path)
    plt.close()


############################################################ Main ############################################################


if __name__ == '__main__':

    #We retrieve the arguments 
    arg = get_parser().parse_args()
    base_model = arg.base_model
    optimized = arg.optimized

    #extraction of the data
    path = os.path.join(data_path,'alpha_dict.npz')
    data = np.load(path)
    data_dict = data['a']
    data_dict = data_dict.reshape(1)
    data_dict = data_dict[0]
    alpha = data_dict[layer][subject]

    create_saving_folder(main_path)

    filename_stimuli = "/home/brain/datasets/SherlockMerlin_ds001110/stimuli/Soundnet_features/sherlock_layer_" + str(layer) + ".npy"
    filename_mask = "/home/brain/datasets/SherlockMerlin_ds001110/sub-" + str(subject) + "/func/sub-" + str(subject) + "_task-SherlockMovie_bold_space-MNI152NLin2009cAsym_brainmask.nii.gz"
    filename_irm = "/home/brain/datasets/SherlockMerlin_ds001110/sub-" + str(subject) + "/func/sub-" + str(subject) + "_task-SherlockMovie_bold_space-MNI152NLin2009cAsym_preproc.nii.gz"

    print('\nData loaded successfully')

    estimator =  Ridge(alpha)
    X_train,X_test,y_train,y_test,masker,meanepi = init(subject,layer,filename_irm,filename_mask,filename_stimuli)
    
    data_path_vector = os.path.join(data_path,'vector_nilearn.npz')
    data_path_meanepi = os.path.join(data_path,'meanepi.nii.gz')
    np.savez_compressed(data_path_vector,a=X_train,b=X_test,c=y_train,d=y_test)
    meanepi.to_filename(data_path_meanepi)
    print('Data saved successfully')

    print('X shape: ',X_test.shape)
    print('Y shape: ',y_train.shape,'\n')

    scores = reference_model_kmeans(X_train,X_test,y_train,y_test,estimator)
    score_img = masker.inverse_transform(scores)

    for num_clust in range(2,11):

        data_path_label = os.path.join(data_path,'layer_'+str(layer),'Label_cluster')
        path = os.path.join(data_path_label,'label_cluster_'+str(num_clust)+'.npz')
        data = np.load(path)
        label = data['a']
        outputpath = os.path.join(main_path,str(num_clust)+'_clustering')
        create_saving_folder(outputpath)
        print('################### Tests for a ',num_clust,' centers clustering ###################')
        for nbc in range(2,num_clust+1):
            print('################ Predicting voxels for cluster number ',nbc,' ################')
            masker = find_cluster(label,masker,data_path,nbc)
            mean = masker.transform(score_img).mean()
            maxi = masker.transform(score_img).max()
            print('Mean: ',mean, 'Max: ', maxi)
            print('')

            if optimized == True:
                #we look for optimizations 
                if num_clust == nbc == 2:
                    best_run, best_model = optim.minimize(model=model_optimization, data=retrieve_data, algo=tpe.suggest, max_evals=5, trials=Trials())
                    X_train,y_train,X_test,y_test = retrieve_data(loaded_stimuli,fmri_ready)
                    print("Evalutation of best performing model:")
                    print(best_model.evaluate(X_test, y_test))
                    print("\nBest performing model chosen hyper-parameters:")
                    print(best_run)
                else:
                    pass
                second_processing(masker,filename_irm,filename_stimuli,outputpath,meanepi,alpha,nbc,num_clust,best_run)
            
            else:

                second_processing(masker,filename_irm,filename_stimuli,outputpath,meanepi,alpha,nbc,num_clust)