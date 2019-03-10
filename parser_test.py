import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd


layer = 16
alpha_test = True
layer_str = 'layer_'+str(layer)
subject_list = [1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17]

outputpath = '/home/brain/matthieu/global_test_output_alpha_lasso'
savingpath = '/home/brain/matthieu/interpretation_opti_log'

def slicing_data(data_dict,key_list):

    #slice the data dictionnary into two dictionnaries where the key in the first one are in the key list
    first_dict = {}
    second_dict = {}

    for key in data_dict:
        if key in key_list:
            first_dict[key] = data_dict[key]
        else:
            second_dict[key] = data_dict[key]

    return first_dict,second_dict

def adjusting_index(df,subject_list):
    df['subjects'] = subject_list
    df = df.set_index('subjects')
    return df


def creating_dataframe(data_dict,subject_list=subject_list):

    df = pd.DataFrame(data_dict)
    print('df =')
    print(df)

    df = adjusting_index(df,subject_list)
    return df



def extracting_data(layer,outputpath,layer_str,alpha_test):

    #we unpack the data for further use (1): creating the path 
    data_path = os.path.join(outputpath,layer_str,'data_for_layer_'+ str(layer)+'.npz')
    
    print('\nStarting interpretation for layer ' + str(layer) + ' ...\n')

    #we unpack the data for further use (2): extraction
    data = np.load(data_path)
    data_dict = data['a']
    data_dict = data_dict.reshape(1)
    data_dict = data_dict[0]

    if alpha_test == False:

        df = creating_dataframe(data_dict)
        return df
    
    else:

        alpha_list = data_dict['alpha_used']

        #we must do a little bit of slicing because the dictionnary didn't work as expected
        index_min = (layer-1)*16
        index_max = index_min+16

        alpha_list = alpha_list[index_min:index_max]
        data_dict ['alpha_used'] = alpha_list

        index_list = []

        for i in range(len(alpha_list)):
            index = {}
            index[subject_list[i]]=alpha_list[i]
            index_list.append(index)

        #we create the  two dataframe 
        alpha_dict,data_dict_final = slicing_data(data_dict,['alpha_used'])
        print('alpha_dict',alpha_dict)

        print()
        df = creating_dataframe(data_dict_final,index_list)
        df_alpha = creating_dataframe(alpha_dict)

        return df,df_alpha



df,df_alpha = extracting_data(layer,outputpath,layer_str,alpha_test)

print(df,df_alpha)