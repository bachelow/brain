
########################################################## READ ME ##################################################################
#                                                                                                                                   #
#   This script use the data from the sherlock experiment and perform a segmentation of the feature vectors for a given layer and   #
#                       a given subject. There is a lot of option (look for them using the -h on the shell)                         #
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

from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.metrics import r2_score

import argparse,os      

################################ Reference model without segmentation ################################

def reference_model(X_train,X_test,y_train,y_test,masker,outputpath,estimator,meanepi):

    estimator.fit(X_train,y_train)
    predictions = estimator.predict(X_test)
    scores = r2_score(y_test, predictions,multioutput='raw_values')
    scores[scores < 0] = 0
    maxref = np.max(scores)
    score_map_img = masker.inverse_transform(scores)
    final_img = threshold_img(score_map_img, threshold=1e-6)

    plot_stat_map(final_img, bg_img=meanepi,cut_coords=5,display_mode='z', aspect=1.25,threshold=1e-6, 
                    title="Results for the reference (no segmentation)")
    #We save the images 
    img4_name = 'ref_img.png'
    img4_path = os.path.join(outputpath,img4_name)
    plt.savefig(img4_path)
    print("Images saved successfully")
    plt.close()

    return maxref


########################################## Important part ############################################ 


def mean_and_reshape(sample_size,nb_voxel,prediction_table):

    #utility function in order to avoid the memory errors

    #it perform a mean of the prediction table (a numpy array of shape (int < 8, nb_voxel)
    mean_prediction = np.mean(prediction_table,axis=0)
    mean_prediction = np.reshape(mean_prediction,(sample_size,nb_voxel))
    return mean_prediction


def estimate_with_segmentation(X_train,X_test,y_train,y_test,subvector_size,masker,estimator):

    feature_size = len(X_train.T)
    nb_voxel = len(y_train.T)
    #begining of the segmentation: checking if the length of the feature vector is a multiple of the subvector size
    try:
        remainder = feature_size % subvector_size
        if remainder != 0:
            raise ValueError("the segmentation can't be performed: length of the feature vector is not a multiple of the subvector size")    
    except ValueError:
        raise 

    #initialisation of the segmentation     
    #the different size which the subvectors will be of given size 
    nb_subvector = feature_size//subvector_size
    sample_size = len(X_train)

    #Vector segmentation (reshape of the feature samples). We swap axis to avoid dimension problems
    X_train = np.reshape(X_train,(sample_size,nb_subvector,subvector_size))
    X_train = X_train.swapaxes(0,1)
    X_test = np.reshape(X_test,(sample_size,nb_subvector,subvector_size))
    X_test = X_test.swapaxes(0,1)
    
    score_table = []
    R2_img_list = []
    prediction_loop_table = []
    prediction_table = []
    
    #main loop: we run our model (fitting and prediction) for each subvectors 
    for curXtrain,curXtest in zip(X_train,X_test):
        
        estimator.fit(curXtrain,y_train)
        predictions=estimator.predict(curXtest)
        scores = r2_score(y_test, predictions,multioutput='raw_values')
        score_table.append(scores)
        prediction_loop_table.append(predictions)

        # we use two list for predictions in order to avoid memory problems, basically if there is more than 8 subvectors
        # we take the mean of those 8 and put it on the second list (what I called here "cleaning")
        if len(prediction_loop_table) == 8:
            print('8 elements in the loop prediction table')
            print('Cleaning prediction table ...')
            submean_prediction = mean_and_reshape(sample_size,nb_voxel,prediction_loop_table)
            prediction_loop_table = []
            prediction_table.append(submean_prediction)
            print('Done')
        else:
            pass

        #for more visibility, we dish out all negative scores, apply the mask defined above and put a threshold at 1e-6 
        scores[scores < 0] = 0
        score_map_img = masker.inverse_transform(scores)
        R2_img_list.append(threshold_img(score_map_img, threshold=1e-6))

    #now we have to deals with all cases (if there are still prediction on both table or just one...)
    if len(prediction_table) > 1:
        if len(prediction_loop_table) >= 1:
            print('Loop prediction table not empty. Cleaning the remaining elements...')
            submean_prediction = mean_and_reshape(sample_size,nb_voxel,prediction_loop_table)
            prediction_loop_table = []
            prediction_table.append(submean_prediction)
            print('Done')
        else:
            pass
        print('Performing mean and reshape on the final predictions ...')
        mean_prediction = mean_and_reshape(sample_size,nb_voxel,prediction_table)
        print('Done')
    elif len(prediction_table) == 1:
        print('Only one element in the prediction table. No operation needed')
        mean_prediction = prediction_table[0]
    else:
        if len(prediction_loop_table) !=0:
            print('Under 8 prediction in total. Performing mean and reshape...')
            mean_prediction = mean_and_reshape(sample_size,nb_voxel,prediction_loop_table)
            print('Done')
        else:
            pass

    #Final processing        
    mean_score = r2_score(y_test,mean_prediction,multioutput='raw_values')
    mean_score[scores < 0] = 0
    mean_score_map_img = masker.inverse_transform(scores)
    mean_img = threshold_img(mean_score_map_img, threshold=1e-6)

    return R2_img_list,score_table,nb_subvector,mean_img,mean_score


########################################## ploting functions #########################################


def ploting_results_image(R2_img_list,exhaustive_mode,meanepi,outputpath,subvector_size,mean_img):
    
    if exhaustive_mode == True:
        #Plot the image for each sub vectors if the exhaustive mode is true
        R2_all = concat_imgs(R2_img_list)

        for number,score_map_img in enumerate(R2_img_list):

            plot_stat_map(score_map_img, bg_img=meanepi, cut_coords=5, display_mode='z', aspect=1.25, threshold=1e-6, 
                title="Results for subvector n°" + str(number+1))
            svg_tempo = 'size_' + str(subvector_size) + '_subvector_' + str(number+1) + '.png'
            img0_path = os.path.join(outputpath,svg_tempo)
            plt.savefig(img0_path)
            #We close the figure each time otherwise only one image will be saved 
            plt.close()

        #We save the images
        img_name = 'img_list_exhaustive_mode' + str(subvector_size) + '.nii.gz'
        img_path = os.path.join(outputpath,img_name)
        R2_all.to_filename(img_path)
    else:
        pass

    #Plot the mean image of all images obtained by the different segmentation  
    plot_stat_map(mean_img, bg_img=meanepi, cut_coords=5, display_mode='z', aspect=1.25, threshold=1e-6,
        title="Mean image")

    #We save the images 
    img2_name = 'mean_img' + str(subvector_size) + '.nii.gz'
    img2_path = os.path.join(outputpath,img2_name)
    mean_img.to_filename(img2_path)
    print("Images saved successfully")
    plt.close()


def ploting_results_max_comparison(score_table,nb_subvector,outputpath,subvector_size,maxref):
 
    #Plot the max score in respect to the different clusters
    allmax = []
    for curscore in score_table:
        print(np.max(curscore))
        allmax.append(np.max(curscore))
    
    x_axis = range(1,nb_subvector+1)
    
    if maxref != 0:
        maxrefline = [maxref for i in x_axis]
        plt.plot(x_axis,allmax,x_axis,maxrefline,'r')
    else:
        plt.plot(x_axis,allmax)

    plt.grid(True)
    plt.xlabel('subvectors')
    plt.ylabel('scores')
    img3_path = os.path.join(outputpath,'plot_max_for_size_' + str(subvector_size) + '.png')
    plt.savefig(img3_path)
    plt.close()
    print("Image saved successfully")


########################################## main functions ############################################


def test_several_values(X_train,X_test,y_train,y_test,meanepi,masker,n_subvector_list,exhaustive_mode,outputpath,maxref,estimator,test_mode):

    #this function is used to test the segmentation for several subdivision 
    print("Conducting tests for the list " + str(n_subvector_list) + " ...")
    saving_path = outputpath
    mean_score_list = []

    for i,subvector_size in enumerate(n_subvector_list):
        
        #we create a special folder for each size
        path_str = 'size_' + str(subvector_size)
        saving_path = os.path.join(outputpath,path_str)
        create_saving_folder(saving_path)

        #we begin the tests 
        print("length of the subvector: " + str(subvector_size))
        R2_img_list,score_table,nb_subvector,mean_img,mean_score = estimate_with_segmentation(X_train,X_test,y_train,y_test,
            subvector_size,masker,estimator)
        ploting_results_image(R2_img_list,exhaustive_mode,meanepi,saving_path,subvector_size,mean_img)
        ploting_results_max_comparison(score_table,nb_subvector,saving_path,subvector_size,maxref)
        mean_score_list.append(mean_score)
        print("Done")

    if test_mode == True:
        return  mean_score_list
    else:
        pass

    print("Over and out")


def check_value():

    check_var = False
    while check_var == False:

        input_str = input("List the value(s) you want to test (if there is more than one, write them separated with a comma, no space): ")
        input_list = list(input_str)
        allowed_chars = '1234567890,'
        i = 0

        while i < len(input_list):
            if input_list[i] in allowed_chars:
                i += 1  
            else:
                print("Invalid syntax: please write the values to test as follow: value1,value2,value3 ...")
                break

        if i == len(input_list):        
            check_var = True
        else:
            pass

    return input_str  


def get_parser():

    #we code the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--default', help = 'Test with the default list of subvector size [4,8,16,32]', 
        action = 'store_true')
    parser.add_argument('-t', '--test', help = 'For conducting test, it just change the storing file', action = 'store_true')
    parser.add_argument('-s', '--subject', help = 'Choose the subject, if not specified it will be subject 3', 
        type = int, default = 3)
    parser.add_argument('-r', '--reference', help = 'Compare with results using no segmentation', action = 'store_true')
    parser.add_argument('-n', '--number', help = 'Specify you want to fill number of subvector instead of its length', 
        action = 'store_true')
    parser.add_argument('-a', '--alpha', help = 'Specify the alpa', type = float, default = 2.5)
    parser.add_argument('-o','--optimal', help = 'Choose the optimal alpha of this subject and layer', action = 'store_true')
    parser.add_argument('-m','--mode', help = 'Enable exhaustive mode', action = 'store_true')
    parser.add_argument('-l', '--layer', help = 'Choose the layer, if not specified it will be layer 13', type = int, default = 13)

    return parser


def convert_number_size(feature_size, number):

    remainder = feature_size % number
    #same test as in line 38
    try:
        if remainder != 0:
            raise ValueError("the segmentation can't be performed: length of the feature vector is not a multiple of the number of subvectors")     
    except ValueError:
        raise 
    subvector_size = feature_size // number
    return subvector_size


def choose_alpha(X_train,X_test,y_train,y_test,masker,outputpath,optimal,test_mode,meanepi):

    #choose the alpha if it is specified in the arguments(either if was specified, or the program choose the optimal alpha), 
    #return the estimator (Ridge(alpha))
    arg = get_parser().parse_args()

    if optimal == True:
        print('Choosing the optimal alpha ...')
        list_alpha = np.logspace(0,5,10)
        n = len(list_alpha)
        list_max = []

        for i in range(n):
            alpha_test = list_alpha[i]
            estimator = Ridge(alpha_test)
            tempo_max = reference_model(X_train,X_test,y_train,y_test,masker,outputpath,estimator,meanepi)
            print('Score for alpha = ' + str(alpha_test) + ': ' + str(tempo_max))
            list_max.append(tempo_max)

        #Now we find the best alpha 
        max_score = max(list_max)
        index = list_max.index(max_score)
        alpha = list_alpha[index]
        print('Choosen alpha: ' + str(alpha))
        print('Done')

        if test_mode == True:
            return Ridge(alpha),alpha
        else:
            pass

    else:
        alpha = arg.alpha
        if test_mode == True:
            return Ridge(alpha),alpha
        else:
            pass

    print('alpha = ' + str(alpha))

    return Ridge(alpha)


def create_saving_folder(outputpath):

    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
        print("New folder created")


def pathfinding(subject,layer,test_mode):

    if test_mode == False:
        #Now we deal with the output file
        #If the path does not exist, we create a directory
        outputpath = '/home/brain/matthieu/test_output'
        sub_path = 'sub_' + subject
        lay_path = 'layer_' + layer
        outputpath = os.path.join(outputpath,sub_path,lay_path) 
        create_saving_folder(outputpath)
    else: 
        outputpath = '/home/brain/matthieu/global_test_output_alpha_lasso'
        lay_path = 'layer_' + layer
        outputpath = os.path.join(outputpath,lay_path)
        create_saving_folder(outputpath)
    
    return outputpath

def init(subject,layer,test_mode):

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
    feature_size = len(X_train.T)

    outputpath = pathfinding(subject,layer,test_mode)

    print('feature_size: ' + str(feature_size))
    print("Init done")


    return X_train,X_test,y_train,y_test,feature_size,masker,meanepi,outputpath


################################################ main ################################################


if __name__ == '__main__':


    #We retrieve the arguments 
    arg = get_parser().parse_args()
    n_subvector_list = [4,8,16,32]

    #Initialize the study according to what was asked:
    comparison_with_ref = arg.reference
    exhaustive_mode = arg.mode
    defaultList = arg.default
    subject = arg.subject
    number_mode = arg.number
    layer = arg.layer  
    optimal = arg.optimal
    test_mode = arg.test

    X_train,X_test,y_train,y_test,feature_size,masker,meanepi,outputpath = init(subject,layer,test_mode)
    estimator = choose_alpha(X_train,X_test,y_train,y_test,masker,outputpath,optimal,test_mode,meanepi)

    #block used only if the user wants to compare with the reference
    if comparison_with_ref == True:
        print("Reference model ...")
        maxref = reference_model(X_train,X_test,y_train,y_test,masker,outputpath,estimator,meanepi)
        print("Over")
    else:
        pass

    if defaultList == True:
        test_several_values(X_train,X_test,y_train,y_test,
            meanepi,masker,n_subvector_list,exhaustive_mode,outputpath,maxref,estimator,test_mode)
    else: 
        #We need to prompt the list of value and check the values (if its prompt as expected, no letters ...)
        #As long as there are errors it keeps asking the user to fill the list of values 
        input_str = check_value()
        n_subvector_list = input_str.split(',')
        for i in range(len(n_subvector_list)):
            n_subvector_list[i] = int(n_subvector_list[i])

        if number_mode == True:
            print("Warning: The list which will be displayed will be the list of the matching sizes, not the numbers")
            for i, number in enumerate(n_subvector_list):
                n_subvector_list[i] = convert_number_size(feature_size,number)
        else:
            pass
        #If there is a value of the subvector length that is not a divisor of the feature vector length 
        #it will be detected in the estimate_with_segmentation function so no need to test
        if comparison_with_ref == True:
            test_several_values(X_train,X_test,y_train,y_test,
                meanepi,masker,n_subvector_list,exhaustive_mode,outputpath,maxref,estimator,test_mode)
        else:
            test_several_values(X_train,X_test,y_train,y_test,
                meanepi,masker,n_subvector_list,exhaustive_mode,outputpath,0,estimator,test_mode)

            
             
            
