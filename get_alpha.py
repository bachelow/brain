import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from nilearn import image,plotting
from nilearn.input_data import NiftiMasker

from nilearn.image import threshold_img
from nilearn.image import mean_img,index_img
from nilearn.image import concat_imgs

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import r2_score,silhouette_score
from sklearn.cluster import dbscan,KMeans

import argparse,os
from tqdm import tqdm 
from check_mask import create_intersect 



subject_list = [1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17]
layer_list = [i for i in range(1,25)]

outputpath = '/home/brain/matthieu/relevant_data'

def create_saving_folder(outputpath):

    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
        print("New folder created")


def reference_model(X_train,X_test,y_train,y_test,masker,outputpath,estimator,meanepi,layer=0,subject=0,worth_saving=False):

    estimator.fit(X_train,y_train)
    predictions = estimator.predict(X_test)
    scores = r2_score(y_test, predictions,multioutput='raw_values')
    scores[scores < 0] = 0
    maxref = np.max(scores)

    if worth_saving == True:
        savingpath = os.path.join(outputpath,'layer_'+str(layer))
        create_saving_folder(savingpath)

        score_map_img = masker.inverse_transform(scores)
        final_img = threshold_img(score_map_img, threshold=1e-6)
        plotting.plot_stat_map(final_img, bg_img=meanepi,cut_coords=5,display_mode='z', aspect=1.25,threshold=1e-6, 
                        title="Results for the reference (no segmentation)")
        
        #We save the images 
        img1_name = 'ref_img_layer'+str(layer)+'_sub_'+str(subject)+'.png'
        img1_path = os.path.join(savingpath,img1_name)
        plt.savefig(img1_path)

        #we save de data of the images
        img2_name = 'ref_img_layer'+str(layer)+'_sub_'+str(subject)+'.nii.gz'
        img2_path = os.path.join(savingpath,img2_name)
        final_img.to_filename(img2_path)

        print("Images saved successfully\n")
        plt.close()

        return None

    return maxref



def test_alpha(X_train,X_test,y_train,y_test,masker,outputpath,meanepi,layer,subject,final_dict):


    print('Choosing the optimal alpha ...')
    list_alpha = np.logspace(0,5,10)
    n = len(list_alpha)
    list_max = []

    for i in tqdm(range(n)):
        alpha_test = list_alpha[i]
        estimator = Ridge(alpha_test)
        tempo_max = reference_model(X_train,X_test,y_train,y_test,masker,outputpath,estimator,meanepi,layer,subject)
        #print('Score for alpha = ' + str(alpha_test) + ': ' + str(tempo_max))
        list_max.append(tempo_max)

    #Now we find the best alpha 
    max_score = max(list_max)
    index = list_max.index(max_score)
    alpha = list_alpha[index]
    print('\nChoosen alpha: ' + str(alpha))
    print('Done')

    #we store the result
    final_dict[layer].append(alpha)

    return Ridge(alpha)

        
def init_alpha(subject,layer,masker):

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
    filename_stimuli = "/home/brain/datasets/SherlockMerlin_ds001110/stimuli/Soundnet_features/sherlock_layer_" + layer + ".npy"
    filename_irm = "/home/brain/datasets/SherlockMerlin_ds001110/sub-" + subject + "/func/sub-" + subject + "_task-SherlockMovie_bold_space-MNI152NLin2009cAsym_preproc.nii.gz"

    meanepi = (index_img(filename_irm,22)) ## instead of calculating the mean epi, we just take one image. 
    loaded_stimuli = np.load(filename_stimuli)
    fmri_data = masker.transform(filename_irm)
    fmri_ready = fmri_data[17:-(fmri_data.shape[0]-17-loaded_stimuli.shape[0])]

    # building the encoding models
    middle = int(loaded_stimuli.shape[0]/2)
    y_train = fmri_ready[:middle] 
    y_test = fmri_ready[middle:]
    X_train = (loaded_stimuli[:middle])
    X_test = (loaded_stimuli[middle:])
    feature_size = len(X_train.T)

    print('feature_size: ' + str(feature_size))
    print("Init done")

    return X_train,X_test,y_train,y_test,feature_size,masker,meanepi


if __name__ == '__main__': 

    create_saving_folder(outputpath)

    print('\nSetting up the intersection mask ...\n')
    mask_inter = create_intersect(subject_list)
    print('\nInitialising the masker with the intersection mask ...\n')
    masker = NiftiMasker(mask_img=mask_inter, detrend=True,standardize=True)
    masker.fit()
    print('Done\n')

    final_dict = {}

    for layer in layer_list:

        #initialize the dict
        final_dict[layer] = []

        for subject in subject_list:
            
            X_train,X_test,y_train,y_test,feature_size,masker,meanepi = init_alpha(subject,layer,masker)
            estimator = test_alpha(X_train,X_test,y_train,y_test,masker,outputpath,meanepi,layer,subject,final_dict)
            if subject >= 14 or layer >=22:   
                reference_model(X_train,X_test,y_train,y_test,masker,outputpath,estimator,meanepi,layer,subject,True)
    
    print(final_dict)
    savingpath = os.path.join(outputpath,'alpha_dict.npz')
    np.savez_compressed(savingpath,a=final_dict)
    print('Data saved successfully')