
########################################################## READ ME ##################################################################
#                                                                                                                                   #
#       Perform a k mean clustering on the array of scores provided by the Ridge estimation for ONE subject and ONE layer           #
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
from nilearn.image import mean_img
from nilearn.image import concat_imgs

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import r2_score,silhouette_score
from sklearn.cluster import dbscan,KMeans

import argparse,os

#################################### Global variables ####################################

layer = 19
estimator = Ridge(alpha=27825)
subject_list = [1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17]
cluster_list = [i for i in range(2,11)]
main_path = '/home/brain/matthieu/test_cluster'

#################################### Utility functions ###################################


def create_saving_folder(outputpath):

    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
        print("New folder created")


#################################### Obtaining scores ####################################


def reference_model(X_train,X_test,y_train,y_test,outputpath,estimator):

    #create the scores array using Ridge and save the images for further comparison 
    estimator.fit(X_train,y_train)
    predictions = estimator.predict(X_test)
    scores = r2_score(y_test, predictions, multioutput='raw_values')
    scores[scores < 0] = 0

    score_map_img = masker.inverse_transform(scores)
    final_img = threshold_img(score_map_img, threshold=1e-6)

    plotting.plot_stat_map(final_img, bg_img=meanepi,cut_coords=5,display_mode='z', aspect=1.25,threshold=1e-6, 
                    title="Results for the reference (no segmentation)")
    #We save the images 
    img4_name = 'ref_img.png'
    img4_path = os.path.join(outputpath,img4_name)
    plt.savefig(img4_path)
    print("Images saved successfully")
    plt.close()

    return scores


def init(subject,layer):

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
    filename_mask = "/home/brain/datasets/SherlockMerlin_ds001110/sub-" + subject + "/func/sub-" + subject + "_task-SherlockMovie_bold_space-MNI152NLin2009cAsym_brainmask.nii.gz"
    filename_irm = "/home/brain/datasets/SherlockMerlin_ds001110/sub-" + subject + "/func/sub-" + subject + "_task-SherlockMovie_bold_space-MNI152NLin2009cAsym_preproc.nii.gz"

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
    feature_size,sample_size = len(X_train.T),len(X_train)
    y_size,y_transp_size = len(y_train),len(y_train.T)

    outputpath = '/home/brain/matthieu/test_cluster/sub_' + subject

    print('\nx vector shape: ',sample_size,feature_size)
    print('y vector shape: ',y_size,y_transp_size,'\n')
    print("Init done\n")

    return X_train,X_test,y_train,y_test,outputpath,meanepi,y_transp_size,masker


#################################### Clustering part #####################################


def find_cluster(scores,nbc,subject):

    #Perform the Kmean algorithm on the scores array in order to finc nbc clusters for 

    n = len(scores)
    scores = scores.reshape(n,1)

    #Data sub-sampling: it avoids the memory errors 
    ind_perm = np.random.permutation(n)[:15000]

    #compute clustering
    #number of cluster choosen
    print('Choosen number of cluster: ',nbc)
    print('Begining clustering ...')

    kmeans = KMeans(n_clusters=nbc,max_iter = 500).fit(scores)
    label = kmeans.predict(scores)

    #show results
    print('Cluster choosen: ')
    print(kmeans.cluster_centers_)

    #silhouette scores (tool to know if the clustering is relevant or not)
    sc = silhouette_score(scores[ind_perm],label[ind_perm], metric='euclidean')
    print('Silhouette score: ',sc)

    unique, counts = np.unique(label, return_counts=True)
        
    print('Cluster size: ')
    print(dict(zip(unique, counts)))

    result_dict[subject].append(sc) 

    return label,nbc


def main_plot(label,meanepi,nbc,outputpath,result_dict,subject):

    #plotting the nifti image 
    score_map_img = masker.inverse_transform(label)
    plt.figure()
    plotting.plot_roi(score_map_img, bg_img=meanepi, title="Results of the clustering", 
        cut_coords = 5, display_mode='z', aspect=1.25)
    img_name = 'Clustering_results.png'
    img_path = os.path.join(outputpath,img_name)
    plt.savefig(img_path)
    print('Image saved successfully')
    plt.close()

    #prepare the data for the lineplot via sns
    df = pd.DataFrame(result_dict[subject])
    df['nb_cluster'] = cluster_list
    df = df.set_index('nb_cluster')
    #plotting the line plot via sns
    sns.lineplot(data=df)
    save_name = 'lineplot_sub'+str(subject)+'.png'
    save_path = os.path.join(outputpath,save_name) 
    plt.savefig(save_path)
    print('Image saved successfully')
    plt.close()


def plot_stats(result_dict,main_path,cluster_list):

    #plot the stats regarding each subject for each type of clustering

    #preparing the data: creating a dataframe [column: subject / row: number of cluster center / data: silouette scores]
    df = pd.DataFrame(result_dict)
    df['cluster_number'] = cluster_list
    df = df.set_index('cluster_number')
    #plotting a swarmplot encapsed into a violin plot (because it is pretty)
    plt.figure(figsize=(10,10))
    sns.violinplot(data=df,inner=None)
    sns.swarmplot(data=df,color='k',alpha=0.8)
    save_name = 'Swarmplot_silhouette.png'
    save_path = os.path.join(main_path,save_name) 
    plt.savefig(save_path)
    print('Image saved successfully')
    plt.close()

    #plotting the lineplot of silouhette scores for each subject wrt the number of cluster choosen
    subjects = df.columns.values
    print(df)
    plt.figure(figsize=(11,10))
    palette_lp = (sns.color_palette("hls",16))
    sns.set() 
    sns.lineplot(data=df,hue=subjects,palette=palette_lp,dashes=False)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    save_namelp = 'lineplot_wrt_subject.png'
    save_pathlp = os.path.join(main_path,save_namelp) 
    plt.savefig(save_pathlp)
    print('Image saved successfully')
    plt.close()


########################################## main ##########################################


if __name__ == '__main__':

    m = len(subject_list)
    result_dict = {}

    for subject in subject_list: 

        result_dict[subject] = []
        X_train,X_test,y_train,y_test,outputpath,meanepi,y_transp_size,masker = init(subject,layer)
        create_saving_folder(outputpath)

        scores = reference_model(X_train,X_test,y_train,y_test,outputpath,estimator) 

        for nbc in cluster_list:
            label,nbc = find_cluster(scores,nbc,subject)

        main_plot(label,meanepi,nbc,outputpath,result_dict,subject)

    plot_stats(result_dict,main_path,cluster_list)

    