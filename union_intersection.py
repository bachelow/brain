
########################################################## READ ME ##################################################################
#                                                                                                                                   #
#   This script was created in order to collect data after applying masks to the predictions and studying the impact of those masks #
#                       there is an option of creating a union mask but it was not relevant for my internship                       #
#                                                                                                                                   #
#####################################################################################################################################


################### Imports ###################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from nilearn import image,plotting,masking
from nilearn.input_data import NiftiMasker

from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.metrics import r2_score

import argparse,os       

from final_segmentation_mask import *
from script_1E import making_size_list

################### Global variables ###################

subject_list = [1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17]
layer_list = [19,20,21,22]
nb_subvector_list = [1,4,8,16,32]
outputpath = '/home/brain/matthieu/test_output_mask'
len_l = len(layer_list)
len_n = len(nb_subvector_list)

################### Utility functions ###################

def get_parser_mask():

    #we code the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--union', help = 'Search the union of the voxel after applying the mask',
        action = 'store_true')
    parser.add_argument('-i', '--intersection', help = 'Search the intersection of the voxel after applying the mask', 
        action = 'store_true')
    parser.add_argument('-c', '--collect', help = 'Collect the masks for all layer between 19 and 22', 
        action = 'store_true')
    parser.add_argument('-s', '--smooth', help = 'choose the 100 biggest values in the score table instead of percentile', 
        action = 'store_true')

    return parser 


def load_np_data(path):

    #extraction of the data
    data = np.load(path)
    data_dict = data['a']
    data_dict = data_dict.reshape(1)
    data_dict = data_dict[0]

    return data_dict


def retrieving_mask(savingpath,use_data):

    #we load either the image or the image and the array coding the image
    if use_data == False:
        img = image.load_img(savingpath)
        return img

    else:
        img = image.load_img(savingpath)
        data = img.get_data()
        unique, counts = np.unique(data, return_counts=True)
        print('Score percentile:')
        print(dict(zip(unique, counts)))
        return data,img


def path_finding_mask(outputpath,subject,layer):

    sub_path = 'sub_' + str(subject)
    lay_path = 'layer_' + str(layer)
    global choice

    #we create the different path regarding if the user want to create an intersection or an union mask
    if intersection == True:
        savingpath_mask = os.path.join(outputpath,'intersection',lay_path,sub_path)
        savingpath_percentile = os.path.join(outputpath,'intersection',lay_path)
        choice = 'i'
        create_saving_folder(savingpath_mask)
        return savingpath_mask,savingpath_percentile,''

    elif union == True:    
        savingpath_mask = os.path.join(outputpath,'union',lay_pat,sub_path)
        savingpath_percentile = os.path.join(outputpath,'union',lay_path)
        choice = 'u'
        create_saving_folder(savingpath_mask)
        return savingpath_mask,savingpath_percentile,''

    elif union == False and intersection == False:
        
        #we use a function of final segmentation mask to check the value 
        if choice == '':
            print('The following data will be used either for creating an intersection mask or a union mask')
            input_str = check_value()
        else:
            input_str = choice

        if input_str == 'i':
            savingpath_mask = os.path.join(outputpath,'intersection',lay_path,sub_path)
            savingpath_percentile = os.path.join(outputpath,'intersection',lay_path)
        else:
            savingpath_mask = os.path.join(outputpath,'union',lay_pat,sub_path)
            savingpath_percentile = os.path.join(outputpath,'union',lay_path)

        create_saving_folder(savingpath_mask)
        return savingpath_mask,savingpath_percentile,input_str


def standart_dict(key_list,value_list,final_dict,subject,index):

    #we create a storing dictionnary for the percentiles / masks the same way:
    #{key --> number of subvectors (int): value --> percentile / mask per subjects (list) }
    #the list is sorted by increasing subject (index 0 --> subject 1, index 1 --> subject 2 ...) 

    key = key_list[index]
    
    #we initialize the key of the final dictionnary if it doesn't exist yet
    if subject == 1:
        final_dict[key] = []
    else: 
        pass

    #This additional test grants the reusable feature for this function
    if len(value_list) != 1:
        value = value_list[index]
    else:
        value = value_list[0]

    final_dict[key].append(value)
    print('Value saved\n')


################### Init function ###################


def init_tests(subject,input_str):

    #this is an additional feature in order to bypass the collect if we have already collected the data
    global choice
    if subject == 1:

        union_tempo,inter_tempo = False,False

        if union == True or input_str == 'u':
            union_tempo = True
            choice = 'u'
        elif intersection == True or input_str == 'i':
            inter_tempo = True
            choice = 'i'

    else:

        if choice == 'i':
            union_tempo,inter_tempo = False,True
        else:
            union_tempo,inter_tempo = True,False

    return union_tempo,inter_tempo


def creating_data(subject,layer,savingpath_mask,savingpath_percentile,input_str):

    global list_dict
    #we collect the data and we conduct the tests
    X_train,X_test,y_train,y_test,feature_size,masker,meanepi = init_mask(subject,layer)
    
    #we save meanepi (back ground image for the roi plot)
    meanepi_path = os.path.join(savingpath_mask,'meanepi_sub_' + str(subject) + '.nii.gz')
    meanepi.to_filename(meanepi_path)

    #we save the data if we want to create masks later 
    size_list,show_list = making_size_list(feature_size,nb_subvector_list,layer)
    list_dict['size_list'] = size_list
    list_dict['show_list'] = show_list
    data_path = os.path.join(savingpath_percentile,'size.npz')
    np.savez_compressed(data_path,a=list_dict)

    #we store the information regarding the futur use of the tests and we conduct those tests 
    union_tempo,inter_tempo = init_tests(subject,input_str)
    percentile_info_list = test_several_values(X_train,X_test,y_train,y_test,meanepi,masker,size_list,
        savingpath_mask,estimator,union_tempo,inter_tempo,smooth)

    print('\nPercentile for the different size:',percentile_info_list,'\n')

    #we store the data regarding the percentiles in a dictionnary, it will be save afterwards in a npz file 
    for index in range(len(show_list)):
        standart_dict(show_list,percentile_info_list,percentile_dict,subject,index)

    print('\npercentile_dict: ',percentile_dict)


################### Storage / data manipulation functions ###################


def storing_masks(size_list,show_list,savingpath_mask,use_data=True):

    for index,size in enumerate(size_list):

        #we unpack the mask first (one by one)
        img_name = 'mean_img_mask_size_' + str(size) + '.nii.gz'
        img_path = os.path.join(savingpath_mask,img_name)
        print('Size ',size,' ...')
        data,img = retrieving_mask(img_path,use_data)
        print('Done')
        tempo = [img]
        
        #we store the current mask into a standardized dictionnary (see line 123) 
        standart_dict(show_list,tempo,mask_dict,subject,index)



def collect_completed(savingpath_percentile,savingpath_mask):

    global list_dict

    #we assume that the collection step has already been completed, 
    #so we go fetch the data where they should be
    list_path = os.path.join(savingpath_percentile,'size.npz')
    list_dict = load_np_data(list_path)
    meanepi_path = os.path.join(savingpath_mask,'meanepi_sub_' + str(subject) + '.nii.gz')
    meanepi = retrieving_mask(meanepi_path,False)

    #now we store each masks in a standardized dictionnary (same architecture as the one for the percentiles)
    #so it will be easy to fetch them afterwards for collection
    show_list = list_dict['show_list']
    size_list = list_dict['size_list']         
    print('Retrieving mask ...\n')
    storing_masks(size_list,show_list,savingpath_mask)

    return meanepi


################### Main functions ###################


def create_intersect_mask(show_list,meanepi,savingpath_percentile,layer,mask_dict,intersection):

    #We create the mask of intersection for all subject for each number of subvectors & each layer  
    for number in show_list:
        mask_list = mask_dict[number]
        #in the doc, threshold = 1 will make the function compute the intersection of masks
        #threshold = 0 will compute the union of all mask in mask list
        if intersection == True:
            mask_inter = masking.intersect_masks(mask_list,threshold=1,connected=False)
        else:   
            mask_inter = masking.intersect_masks(mask_list,threshold=0,connected=False)

        #We save the mask 
        if intersection == True:
            mask_path = os.path.join(savingpath_percentile,
                'mask_intersection_layer_' + str(layer) + '_number_' + str(number) + '.nii.gz')
        elif union == True:
            mask_path = os.path.join(savingpath_percentile,
                'mask_union_layer_' + str(layer) + 'number' + str(number) + '.nii.gz')
        mask_inter.to_filename(mask_path)

        #we plot the results 
        plotting.plot_roi(mask_inter,title='roi',bg_img = meanepi)


def plot_final_figure(len_n,len_l,layer,savingpath_percentile,subject_list,fig):

    global choice

    #we define the path that will be used 
    if intersection == True or choice == 'i':
        data_path = os.path.join(savingpath_percentile,'percentile_intersection_layer_' + str(layer))
        input_str = 'mask_intersection_layer_'

    elif union == True or choice == 'u':
        data_path = os.path.join(savingpath_percentile,'percentile_union_layer_' + str(layer))
        input_str = 'mask_union_layer_'

    else:
        pass

    data_dict = load_np_data(data_path + '.npz')

    #transformation into a data frame 
    df = pd.DataFrame(data_dict)
    df['subjects'] = subject_list
    df = df.set_index('subjects')

    #we retrieve the background image
    meanepi_path = os.path.join(savingpath_percentile,'sub_1','meanepi_sub_1.nii.gz')
    meanepi = retrieving_mask(meanepi_path,False)

    i = layer_list.index(layer)
    plt.figure(fig.number)

    #we plot the intersection mask of all subject for each subdivision (see nb_cluster_list)
    #this is for all layer in the layer list so the different subplot are organized as follow:
    #each row represents a given layer. Each column represent a given subdivision   
    for j in range(len_n):
        size_s = str(nb_subvector_list[j])
        path = os.path.join(savingpath_percentile,input_str + str(layer) + '_number_' + size_s + '.nii.gz')
        mask_inter = retrieving_mask(path,False)
        ax = plt.subplot(len_l,len_n+1,6*i+j+1)
        plotting.plot_roi(mask_inter,bg_img=meanepi,axes=ax,figure=fig)

    plt.subplot(len_l,len_n+1,6*i+len_n+1)    
    sns.swarmplot(data=df)


################################################    Main    ################################################


if __name__ == '__main__':


    arg = get_parser_mask().parse_args()

    #we initialyse the global variables / other variables
    union = arg.union
    intersection = arg.intersection
    collect = arg.collect
    smooth = arg.smooth
    estimator = Ridge(alpha=2.5)
    percentile_dict = {}
    mask_dict = {}
    
    #the list dict is a global variable if the user wants to create the data + the mask at once
    #otherwise the list dict is saved and is reused later 
    global choice,list_dict
    choice = ''
    list_dict = {}
    
    plot_fig = False
    print(len_l,len_n)

    fig = plt.figure(figsize=(20,15))

    for layer in layer_list:
        for subject in subject_list:

            #we begin by creating the saving folder for the matching type of mask 
            savingpath_mask,savingpath_percentile,input_str = path_finding_mask(outputpath,subject,layer)

            #we create the data if there is none
            if collect == True:
                print('Creating mask ...')
                creating_data(subject,layer,savingpath_mask,savingpath_percentile,input_str)

                #we save the data regarding the percentile used
                if intersection == True or choice == 'i':
                    data_path = os.path.join(savingpath_percentile,'percentile_intersection_layer_' + str(layer))
                elif union == True or choice == 'u':
                    data_path = os.path.join(savingpath_percentile,'percentile_union_layer_' + str(layer))
                
                np.savez_compressed(data_path,a=percentile_dict)
                print('Percentile data saved successfully')
            else:
                pass
                
            meanepi = collect_completed(savingpath_percentile,savingpath_mask)

        show_list = list_dict['show_list']
        size_list = list_dict['size_list'] 

        #we create the intersection mask and we save the image
        create_intersect_mask(show_list,meanepi,savingpath_percentile,layer,mask_dict,intersection) 
        plot_final_figure(len_n,len_l,layer,savingpath_percentile,subject_list,fig)
        print('Layer added to the final plot')
        plot_fig = True


    if intersection == True:
        use_str = 'intersection'
    else:
        use_str = 'union'

    if plot_fig == True:
        savename = os.path.join(outputpath,use_str,'final_fig.png')
        plt.savefig(savename)
        print('Image saved successfully')
    else:
        pass

    plt.close()



