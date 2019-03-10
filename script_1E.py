
########################################################## READ ME ##################################################################
#                                                                                                                                   #
#   This script is a thorough investigation of the segmentation method application for all subjects / layers for this experiment    #
#       It will use the segmentation method and then store the data for further usage (see interpretation_result.py file)           #
#                                                                                                                                   #
#####################################################################################################################################

################### Imports ###################

import numpy as np
import matplotlib.pyplot as plt

from nilearn import image
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_epi
from nilearn.plotting import plot_stat_map
from nilearn.image import threshold_img
from nilearn.image import mean_img
from nilearn.image import concat_imgs

from sklearn.linear_model import Ridge, ElasticNet,Lasso
from sklearn.metrics import r2_score

import argparse,os   

from final_segmentation import *

################### Global variables ###################

subject_list = [1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17]
layer_list = [i for i in range(1,25)]
nb_subvector_list = [1,2,4,8,16,32]



    
def making_size_list(feature_size,nb_subvector_list,layer):

    test_list = []
    show_list = []
    passing_loop = False
    for i, number in enumerate(nb_subvector_list):
        if passing_loop == False: 
            size = feature_size // number
            if feature_size >= number:
                test_list.append(size)
                show_list.append(nb_subvector_list[i])
            else: 
                print('Warning: size of the subvectors too great (' + str(number) + ' elements array for a feature size of ' + str(feature_size) + ')')
                print('exiting the loop')
                passing_loop = True
        else:
            pass

    print('Testing the following number of subvectors: ' + str(show_list))
    return test_list,show_list


if __name__ == '__main__':

    arg = get_parser().parse_args()

    #Initialize the study according to what was asked:
    optimal = arg.optimal
    test_mode = arg.test
    exhaustive_mode = arg.mode
    alpha = arg.alpha
    alpha_list = []

    for layer in layer_list:

        plot_mean_dict = {}

        for subject in subject_list:

            #we initialize everything
            X_train,X_test,y_train,y_test,feature_size,masker,meanepi,outputpath = init(subject,layer,test_mode)
            estimator,alpha_test = choose_alpha(X_train,X_test,y_train,y_test,masker,outputpath,optimal,test_mode,meanepi)

            #we save the value of alpha if we used the optimal option
            alpha_list.append(alpha_test)

            #we convert the nb subvector list, and watch out for the size to avoid errors 
            size_list,show_list = making_size_list(feature_size,nb_subvector_list,layer)

            #now we initialize the plot_mean_list for further use because we know exactly the length of the feature vector
            if subject == 1:
                for i in range(len(show_list)):
                    key = show_list[i]
                    plot_mean_dict[key] = []
            else: 
                pass

            #Now we collect the mean R² score for each subvector number, and take the maximum  
            mean_score_list = test_several_values(X_train,X_test,y_train,y_test,
                meanepi,masker,size_list,exhaustive_mode,outputpath,0,estimator,test_mode)

            mean_max_list = []
            for i in range(len(show_list)):
                maximum = np.max(mean_score_list[i])
                mean_max_list.append(maximum)

            #we fill the plot mean list: each sub list contain all R² mean score for 
            #each subjects for one specific number of subvecors
            for i in range(len(mean_max_list)):
                value = mean_max_list[i]
                print('Current values of mean for ' + str(show_list[i]) + ' subvectors: ' + str(value))
                plot_mean_dict[show_list[i]].append(value)

            if optimal == True:
                key = 'alpha_used'
                plot_mean_dict[key] = alpha_list

            print('Test finished. Moving to the next subject...')

        print('No more subject for this layer')
        #now we prepare the plot
        print("Plotting the mean and the error...")
        
        mean_list = []
        var_list = []
        for i in range(len(show_list)):
            data_list = plot_mean_dict[show_list[i]]
            data_path = os.path.join(outputpath,'data_for_layer_' + str(layer))
            np.savez_compressed(data_path,a=plot_mean_dict)
            mean = np.mean(data_list)
            mean_list.append(mean)  
            var = np.std(data_list)
            var_list.append(var)
        
        plt.figure()
        description = 'Layer' + str(layer)
        plt.errorbar(show_list,mean_list,var_list, label=description)
        plt.xlabel('Number of subvectors')
        plt.ylabel('Score: mean and standart deviation')
        plt.legend(loc='best')
        #we save the figure on the right place
        savingpath = outputpath
        img3_path = os.path.join(savingpath,'final_figure_layer_' + str(layer) + '.png')
        plt.savefig(img3_path)
        plt.close()

        print(plot_mean_dict)
        print("Image saved successfully")
        print('Over and out')


            


