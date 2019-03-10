#goal : have all subjects with the same number of voxels, aligned, in MNI space (Non linear)
#
#Step 1 : 
#generate a intersection across subjects of all masks :     filename_mask = "/home/brain/datasets/SherlockMerlin_ds001110/
#sub-" + subject + "/func/sub-" + subject + "_task-SherlockMovie_bold_space-MNI152NLin2009cAsym_brainmask.nii.gz"
#
#Step 2 : check visually that the intersection is ok, for each subject : 
#- calculate the mean epi
#- plot_roi(mask_intersected, bg_img = mean_epi)
#- plot_roi(mask_subject,bg_img = mean_epi)
#-> 17 figures
#-> check the volume occupied by the mask wrt subject
#
#
#Step 3 : use the result of this intersection when estimating Ridge

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from nilearn import image,plotting,masking
from nilearn.input_data import NiftiMasker
from nilearn.image import mean_img,index_img
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.metrics import r2_score
import argparse,os 
from tqdm import tqdm     


subject_list = [1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17]
savingpath = '/home/brain/matthieu/relevant_data'

def create_intersect(subject_list):

    fig = plt.figure(figsize=(20,15))
    mask_list = []
    i = 1 

    for subject in tqdm(subject_list):

        if subject < 10:
            subject_str = '0' + str(subject)
        else:
            subject_str = str(subject)

        filename_mask = "/home/brain/datasets/SherlockMerlin_ds001110/sub-" + subject_str + "/func/sub-" + subject_str + "_task-SherlockMovie_bold_space-MNI152NLin2009cAsym_brainmask.nii.gz"
        filename_irm = "/home/brain/datasets/SherlockMerlin_ds001110/sub-" + subject_str + "/func/sub-" + subject_str + "_task-SherlockMovie_bold_space-MNI152NLin2009cAsym_preproc.nii.gz"

        meanepi = (index_img(filename_irm,22))
        mask_list.append(filename_mask)
        ax = plt.subplot(3,6,i)
        plotting.plot_roi(filename_mask,bg_img=meanepi,axes=ax,figure=fig)
        i+=1

    mask_inter = masking.intersect_masks(mask_list,threshold=1,connected=False)
    ax = plt.subplot(3,6,i)
    plotting.plot_roi(mask_inter,bg_img=meanepi,axes=ax,figure=fig)
    saving_path = os.path.join(savingpath,'mask_inter.nii.gz')
    mask_inter.to_filename(saving_path)
    return mask_inter



if __name__ == '__main__':

    mask_int = create_intersect(subject_list)
    mask_inter.to_filename(savingpath)
    plt.show()
    