
########################################################## READ ME ##################################################################
#                                                                                                                                   #
#   This script use the kmean algorithm in order to perform a clustering on a numpy array containing the scores objtained by Ridge  #
#   regression for evey subject for a given layer (this is a 2 dim clustering where the input verctor has a shape of (nb subject,   #
#   nb_voxels). It calculate the silouhette scores for each layer and each clustering type (whether we have 2,3,4,5 centers ...)    #
#                                                                                                                                   #
#####################################################################################################################################

################### Imports ###################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from nilearn import image,plotting
from nilearn.input_data import NiftiMasker

from nilearn.image import threshold_img
from nilearn.image import index_img

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score,silhouette_score
from sklearn.cluster import KMeans

import os
from tqdm import tqdm 
from check_mask import create_intersect 

################### Global variables ###################

subject_list = [1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17]
layer_list = [20]
cluster_list = [i for i in range(2,11)]
main_path = '/home/brain/matthieu/test_cluster'
data_path = '/home/brain/matthieu/relevant_data'


#################################### Utility functions ###################################

def create_saving_folder(outputpath):

    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
        print("New folder created")

#################################### Obtaining scores ####################################

def reference_model_kmeans(X_train,X_test,y_train,y_test,outputpath,estimator,masker,subject):

    estimator.fit(X_train,y_train)
    predictions = estimator.predict(X_test)
    scores = r2_score(y_test, predictions, multioutput='raw_values')
    scores[scores < 0] = 0

    score_map_img = masker.inverse_transform(scores)
    final_img = threshold_img(score_map_img, threshold=1e-6)

    plotting.plot_stat_map(final_img, bg_img=meanepi,cut_coords=5,display_mode='z', aspect=1.25,threshold=1e-6, 
                    title="Results for the reference (no segmentation)")
    #We save the images 
    img4_name = 'ref_img_sub_'+str(subject)+'.nii.gz'
    img4_path = os.path.join(outputpath,img4_name)
    final_img.to_filename(img4_path)
    print("Images saved successfully")
    plt.close()

    return scores


def init(subject,layer,mask_inter):

    #we transform the layer and subject variables in order to use them in the input path 
    if subject < 10:
        subject = '0' + str(subject)
    else:
        subject = str(subject)
    
    if layer < 10:
        layer = '0' + str(layer)
    else:
        layer = str(layer)

    print("\nInitializing tests for subject " + subject + " and layer " + layer + " ...")
    #we dig out our data from these files and we create the mask in order to have a better rendering in the future plots
    filename_stimuli = "/home/brain/datasets/SherlockMerlin_ds001110/stimuli/Soundnet_features/sherlock_layer_" + layer + ".npy"
    filename_irm = "/home/brain/datasets/SherlockMerlin_ds001110/sub-" + subject + "/func/sub-" + subject + "_task-SherlockMovie_bold_space-MNI152NLin2009cAsym_preproc.nii.gz"

    #initialize the encoding model 
    meanepi = (index_img(filename_irm,22)) ## instead of calculating the mean epi, we just take one image. 


    print('\nInitialising the masker with the intersection mask ...\n')
    masker = NiftiMasker(mask_img=mask_inter, detrend=True,standardize=True)
    masker.fit()
    print('Done')

    loaded_stimuli = np.load(filename_stimuli)
    fmri_data = masker.transform(filename_irm)
    fmri_ready = fmri_data[17:-(fmri_data.shape[0]-17-loaded_stimuli.shape[0])]

    # building the encoding models
    middle = int(loaded_stimuli.shape[0]/2)
    y_train = fmri_ready[:middle] 
    y_test = fmri_ready[middle:]
    X_train = (loaded_stimuli[:middle])
    X_test = (loaded_stimuli[middle:])

    outputpath = main_path + '/lay_' + layer

    print("Init done\n")

    return X_train,X_test,y_train,y_test,outputpath,meanepi,masker


#################################### Clustering part #####################################


def find_cluster(scores,cluster_list,nbc,subject,data_path):

    n = len(scores)

    #Data sub-sampling: it avoids the memory errors 
    ind_perm = np.random.permutation(n)[:15000]

    #compute clustering
    kmeans = KMeans(n_clusters=nbc,max_iter = 500).fit(scores)
    label = kmeans.predict(scores)

    #we add 1 to the labels so the first label is not mistaken with the background. 
    label += 1

    #silhouette scores  
    sc = silhouette_score(scores[ind_perm],label[ind_perm], metric='euclidean')
    sil_dict[nbc].append(sc)
    savingpath = os.path.join(data_path,'Label_cluster','label_cluster_'+str(nbc)+'.npz')
    np.savez_compressed(savingpath,a=label)   

    return label,nbc


def processing_scores(len_list,result_list):

    #we find the array which has the most element and we fill the other array with zeroes in order to have matching size
    maxi = max(len_list)
    
    for scores,length in tqdm(zip(result_list,len_list)):
        
        #we calculate the difference, if its the vector with the 
        #maximum length then we do nothing
        difference = maxi - length
        n = len(scores)
        #print('\nDifference: ',difference)
        if difference == 0:
            #print('No padding needed')
            scores = np.reshape(scores,(difference+n,1))
            result_list.append(scores)
        else:
            print('Proceed to zero padding...')
            list_tempo = [0]*difference
            scores = scores.tolist()
            #now we concatenate the lists and we reshape them into arrays of suitable shape 
            scores += list_tempo
            scores = np.asarray(scores)
            scores = np.reshape(scores,(difference+n,1))
            print(scores.shape)
            print('Zero padding done')
            result_list.append(scores)

    #now we have a result list which contains 2 scores per subject, we sclice it in half
    n = len(result_list)//2
    result_list = result_list[n:]

    result_array = result_list[0]
    for i in range(1,len(result_list)):
        result_array = np.concatenate((result_array,result_list[i]),axis=1)

    return result_array


#################################### Making nice plots ###################################


def main_plot(label,meanepi,nbc,outputpath,sil_dict,subject,masker):

    #plotting the nifti image 
    score_map_img = masker.inverse_transform(label)
    plt.figure()
    plotting.plot_roi(score_map_img, bg_img=meanepi, title="Results of the clustering", 
        cut_coords = 5, display_mode='z', aspect=1.25)
    img_name = 'Clustering_results'+str(nbc)+'.png'
    img_path = os.path.join(outputpath,img_name)
    plt.savefig(img_path)
    #print('Image saved successfully')
    plt.close()


def plot_stats(sil_dict,main_path,layer_list):

    #we create the data frame for futur plots
    df = pd.DataFrame(sil_dict)
    df['Layer list'] = layer_list
    df = df.set_index('Layer list')
    n = len(layer_list)
    #we plot the violin plot of the silouhette scores wrt different number of cluster
    #for all layer in the layer list, the swarm plot will be 
    plt.figure(figsize=(12,10))
    sns.violinplot(data=df,inner=None)
    sns.swarmplot(data=df,color='k',alpha=0.8)
    save_name = 'Swarmplot_silhouette.png'
    save_path = os.path.join(main_path,save_name) 
    plt.savefig(save_path)
    print('Image saved successfully')
    plt.close()

    df = df.T
    print(df)

    #plotting the lineplot of silouhette scores for each layer wrt the number of cluster choosen
    plt.figure(figsize=(11,10))
    palette_lp = (sns.color_palette("hls",n))
    sns.set() 
    sns.lineplot(data=df,palette=palette_lp,dashes=False)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    save_namelp = 'lineplot_wrt_layer.png'
    save_pathlp = os.path.join(main_path,save_namelp) 
    plt.savefig(save_pathlp)
    print('Image saved successfully')
    plt.close()


########################################## main ##########################################


if __name__ == '__main__':

    #setting up the global variables
    sil_dict = {}

    for nbc in cluster_list:
        sil_dict[nbc] = []

    #we set up the intersection mask of every subject in order to predict brain activity on the same voxels      
    print('\nSetting up the intersection mask ...\n')
    mask_inter = create_intersect(subject_list)
    print('Done')
    
    #extraction of the data
    path = os.path.join(data_path,'alpha_dict.npz')
    data = np.load(path)
    data_dict = data['a']
    data_dict = data_dict.reshape(1)
    data_dict = data_dict[0]
    print('\nData load successfully')

    for layer in layer_list: 

        result_list = []
        len_list = []

        for i,subject in enumerate(subject_list): 

            X_train,X_test,y_train,y_test,outputpath,meanepi,masker = init(subject,layer,mask_inter)
            create_saving_folder(outputpath)

            #now we retrieve the optimal alpha
            alpha = data_dict[layer][i]
            print('Alpha used: ',alpha)
            estimator = Ridge(alpha) 
            
            #we obtain the scores 
            scores = reference_model_kmeans(X_train,X_test,y_train,y_test,outputpath,estimator,masker,subject)
            m = len(scores)
            len_list.append(m)            
            result_list.append(scores)

        #now we build the vector for further clustering     
        result_array = processing_scores(len_list,result_list)

        print('\nBeginning the tests for different number of clusters ...\n')
        for nbc in tqdm(cluster_list):
            #we make different clustering (the number of center is given by nbc) and we plot the silhouette score 
            label,nbc = find_cluster(result_array,cluster_list,nbc,subject,data_path)
            main_plot(label,meanepi,nbc,outputpath,sil_dict,subject,masker)

    print(sil_dict)
    plot_stats(sil_dict,main_path,layer_list)

    
