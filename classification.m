clear all
close all
load('finalMat.mat')
[trainedClassifier, validationAccuracy] = linearClassifier(FinalMat) %linear SVM

save('trainedClassifier.mat')

