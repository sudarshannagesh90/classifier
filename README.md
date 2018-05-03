# classifier
Train up a classifier and use it as a feature extractor

##Prerequisites: Install Torch 

##Usage: applyFeatureTransformation(img, classifier_name) 
##converts an image in numpy format to a different feature space which is easily identifiable for an image classifier

##ipynb file contesnts: (put the following in an ipynb file )

%matplotlib inline
##sample function to test the performance of perceptual loss functions
from featureExtractor import applyFeatureTransformation
import cv2
import matplotlib.pyplot as plt

classifier_name = "vggClassifier67"  # can be one of simpleClassifier42 or simpleClassifier58 or vggClassifier67
##Definitions of the classifiers in the functions simpleClassifier.py and vggClassifier.py 

img = cv2.imread('apple.png',0)
plt.imshow(img, cmap='gray')

featureSpaceimg = applyFeatureTransformation(img, classifier_name)
plt.imshow(featureSpaceimg, cmap='gray')
plt.colorbar()



