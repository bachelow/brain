
########################################################## READ ME ##################################################################
#                                                                                                                                   #
#   This script use the data collected from script1E and allow the user to have a graphic interpretation of the results (like swar  #
#   -mplot, violin plot or lineplot, all done via sns and panda). There are several options in order to have different plots        #
#                                                         possible                                                                  #
#                                                                                                                                   #
#####################################################################################################################################

################### Imports ###################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from nilearn import image
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_epi
from nilearn.plotting import plot_stat_map
from nilearn.image import threshold_img
from nilearn.image import mean_img
from nilearn.image import concat_imgs

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import r2_score

import argparse,os      

################### Global variables ###################

outputpath = '/home/brain/matthieu/global_test_output_alpha_lasso'
savingpath = '/home/brain/matthieu/interpretation_opti_log'
subject_list = [1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17]
layer_list = [i for i in range(1,25)]
nb_subvector_list = [1,2,4,8,16,32]
n = len(layer_list)

################### Utility functions ###################

def create_saving_folder(outputpath):

    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
        print("New folder created")

def get_parser():

    #we code the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--everything', help = 'Save each figure on a big subplot', action = 'store_true')
    parser.add_argument('-v', '--violin', help = 'Save swarmplot encapsed within a violinplot', action = 'store_true')
    parser.add_argument('-a', '--alpha', help = 'Indicate you use the data with optimized alpha', action = 'store_true')
    
    return parser

#################### Plot functions ####################

def makingswarmplot(df,savingpath,layer,layer_str):

    if everything == False:

        plt.figure('intermediate')
        save_folder = os.path.join(savingpath,layer_str)
        create_saving_folder(save_folder)    

        if encapsed == True:
            sns.violinplot(data=df,inner=None)
            sns.swarmplot(data=df,color='k',alpha=0.8)
            save_name = os.path.join(save_folder,'violinplot_' + layer_str + '.png')
        else:   
            sns.swarmplot(data=df)
            save_name = os.path.join(save_folder,'swarmplot_' + layer_str + '.png')

        plt.ylabel('R² scores')
        plt.xlabel('Number of subvectors')
        plt.savefig(save_name)
        print('\nSwarmplot saved successfully\n')

        plt.close('intermediate')

    else:
        #now we plot everything into one subplot
        plt.subplot(6,4,layer)
        sns.swarmplot(data=df)
        save_name = os.path.join(savingpath,'swarmplot.png')
        print('\nSubplot added\n')


def makingboxplot(df,savingpath,layer,layer_str):

    if everything == False:

        plt.figure('intermediate')
        save_folder = os.path.join(savingpath,layer_str)
        create_saving_folder(save_folder)    

        sns.boxplot(data=df)
        save_name = os.path.join(save_folder,'boxplot_' + layer_str + '.png')
        plt.ylabel('R² scores')

        plt.xlabel('Number of subvectors')
        plt.savefig(save_name)
        print('\nBoxplot saved successfully\n')

        plt.close('intermediate')

    else:
        #now we plot everything into one subplot
        plt.subplot(6,4,layer)
        sns.boxplot(data=df)
        save_name = os.path.join(savingpath,'boxplot.png')
        print('\nSubplot added\n')
   

def makingheatmap(df,savingpath,layer,layer_str,fig):

    if everything == False:

        plt.figure('intermediate')
        save_folder = os.path.join(savingpath,layer_str)
        create_saving_folder(save_folder)    

        sns.heatmap(df,vmin=0.005,vmax=0.4)
        save_name = os.path.join(save_folder,'heatmap_' + layer_str + '.png')
        plt.ylabel('Subjects')
        plt.tight_layout()
        plt.xlabel('Number of subvectors')
        plt.savefig(save_name)
        print('\nHeatmap saved successfully\n')

        plt.close('intermediate')

    else:
        #now we plot everything into one subplot
        plt.figure(fig.number)
        ax = plt.subplot(6,4,layer)
        sns.heatmap(df,vmin=0.005,vmax=0.4,cbar=False,axes=ax)
        print('\nSubplot added\n')
        
#################### Main plot function (using the 3 aboves) ####################

def plot_stats(df,savingpath,layer,layer_str):
    
    #either do a big figure with subplot for each subject / layer if everything == True

    if everything == True:

        plt.figure('swarmplot',figsize=(15,15))
        save_name_sp = os.path.join(savingpath,'swarmplot.png')
        makingswarmplot(df,savingpath,layer,layer_str)
        if layer == 24:
            plt.tight_layout()
            plt.savefig(save_name_sp)
            plt.close('swarmplot')

        plt.figure('boxplot',figsize=(15,15))
        save_name_bp = os.path.join(savingpath,'boxplot.png')
        makingboxplot(df,savingpath,layer,layer_str)
        if layer == 24:
            plt.tight_layout()
            plt.savefig(save_name_bp)
            plt.close('boxplot')

        fig = plt.figure('heatmap',figsize=(15,15))
        save_name_hm = os.path.join(savingpath,'heatmap.png')
        makingheatmap(df,savingpath,layer,layer_str,fig)
        if layer == 24:
            plt.tight_layout()
            plt.savefig(save_name_hm)
            plt.close('heatmap')

    #if it is not the case it saves a single figure (heatmap / swarmplot / boxplot) for each subject for each layer
    else:
        fig = None 
        makingheatmap(df,savingpath,layer,layer_str,fig)
        makingboxplot(df,savingpath,layer,layer_str)
        makingswarmplot(df,savingpath,layer,layer_str)

#################### Data manipulation function ####################

def adjusting_index(df,subject_list):

    #set the index of the dataframe df as subject_list (because there is some inconsistences in the subject list) 
    df['subjects'] = subject_list
    df = df.set_index('subjects')
    return df


def creating_dataframe(data_dict,subject_list=subject_list):

    #create a pandas dataframe
    df = pd.DataFrame(data_dict)
    df = adjusting_index(df,subject_list)
    return df


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

        df = creating_dataframe(data_dict_final,index_list)
        df_alpha = creating_dataframe(alpha_dict)

        return df,df_alpha


def prep_lineplot(stat_df):

    #store the mean of the dataframe stat_df for further usage
    mean = stat_df.iloc[[1]]
    mean_list.append(mean)


def makinglineplot(mean_list,layer_list,savingpath):

    df_mean = pd.concat(mean_list)
    df_mean = adjusting_index(df_mean,layer_list)
    print('\nFinal plot ...\n')
    
    sns.set()
    sns.lineplot(data=df_mean)
    plt.xlabel('layer')
    plt.xlim(xmin=0)
    plt.ylabel('R² scores')
    save_name_lp = os.path.join(savingpath,'lineplot.png')
    plt.savefig(save_name_lp)
    plt.close()


def main_loop(df,savingpath,layer,layer_str):

    print('\nDisplaying the data for layer ' + str(layer) + ' ...\n')
    print(df)

    #we display the basic statistical figures on the terminal
    stat_df = df.describe()
    print('\nDisplaying the statistics for layer ' + str(layer) + ' ...\n')
    print(stat_df)
    prep_lineplot(stat_df)

    #we plot the figures for each layers
    plot_stats(df,savingpath,layer,layer_str)


def plot_alpha(df_alpha,layer,layer_str,savingpath):

    #Plot the alpha used for each subject within a given layer 
    plt.figure()
    sns.lineplot(data=df_alpha)
    plt.xlabel('subjects')
    plt.ylabel('alpha')
    save_name_lpa = os.path.join(savingpath,layer_str,'lineplot_alpha_layer_' + str(layer) +'.png')
    plt.savefig(save_name_lpa)
    plt.close()



if __name__ == '__main__':

    arg = get_parser().parse_args()
    everything = arg.everything
    encapsed = arg.violin
    alpha_test = arg.alpha
    mean_list = []

    for layer in layer_list:

        if layer < 10:
            layer_str = 'layer_0' + str(layer)
        else:
            layer_str = 'layer_' + str(layer)

        if alpha_test == False:
            df = extracting_data(layer,outputpath,layer_str,alpha_test)
            main_loop(df,savingpath,layer,layer_str)            

        else:
            outputpath = '/home/brain/matthieu/global_test_output_alpha_lasso'
            savingpath = '/home/brain/matthieu/interpretation_opti'
            create_saving_folder(outputpath)

            df,df_alpha = extracting_data(layer,outputpath,layer_str,alpha_test)
            main_loop(df,savingpath,layer,layer_str)
            plot_alpha(df_alpha,layer,layer_str,savingpath)

            print(df)
            print('\n')
            print(df_alpha)

    plt.close()
    makinglineplot(mean_list,layer_list,savingpath)


