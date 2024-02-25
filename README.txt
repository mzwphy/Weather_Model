############################################################
############################################################
############################################################



SOLAR FLARE CATEGORIZATION (CLASSIFICATION)
###########################################


*   A deep Convolutional Neural Network is trained as an attempt
    to aid in space weather forecasting efforts.


*   This multi_class categorization model is trained to predict solar
    flares using magnetograms of Solar Active Regions from the 
    Solar Dynamic Obsevatory (SDO).


*   There are misclassifications but the model can predict the occurance 
    of Solar Flares with 94% accuracy. It has four categories:


        *   No Flare

        *   Category C flare

        *   Category M flare

        *   Category X flare

*   A simple version of this model is also available. 


*   This simple version is a binary classifier, only concerned with whether
    the Solar Active Region is flaring or non-flaring, that is, whether
    there is a solar flare or there is no solar flare.


##############################################################
##############################################################
##############################################################


DATA
####


*  The data used in the training and testing of this model consists of
   about 12000 grayscale images of the sun's active regions with equal contribution
   from all four categories (classes).

* 

#################################################################
#################################################################
#################################################################



COPYING REPOSITORY
##################

*   Make sure you have git installed

*   open ubuntu terminal and type:
       git clone <https link>

*   To run the model and make predictions, navigate to the <run> directory (folder)
    and type "python flarePredictions.py" 

*   Each time you run the script, you will get prediction for one magnetogram. There 
    are 1543 images in the test dataset. It is possible to modify this script to make 
    predictions on custom images but images must be grayscale, 224 by 224 pixels, in array 
    form and rescaled (normalized) to 1/255.



######################################################################
######################################################################
######################################################################


POSSIBLE IMPROVEMENTS
#####################


*   It is possible to improve on the model perfomance by using Transfer Learning 
    techniques, that is, to use the knowledge (weights) of a pre_trained deep learning model
    to train this model. This may show some improvements on the accuracy and overall perfomance
    of the model in predicting flares and Coronal Mass Ejectionss.


*   The strength of each class can also me determined for example, by making use of a 
    regression model. A regression model can tell us how strong the flare is in each 
    category.
