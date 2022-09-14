from cmath import sqrt
import pickle
from turtle import distance
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.spatial import distance


def class_acc(pred,gt):
    """Calculates the class accuracy of predicted- and truth tables"""

    if(len(pred) != len(gt)):
        return 0

    # Makes a list index values, where pred and gt have same values
    # and calculates the list length.
    correct_predictions = len(np.arange(len(pred))[pred==gt])

    return correct_predictions/len(gt)

def cifar10_classifier_random(x):
    """Returns a random list of image label indexes based on input data x"""
    return np.array([random.randrange(9) for i in range(len(x))])

def cifar10_classifier_1nn(x,trdata,trlabels):
    """Returns best matching training data labels based on eucliden distance."""
    baIndex = 0
    bestDistance = -1

    for i in range(0,len(trdata)):

        dist = np.sqrt(np.abs(np.sum(np.power(x - trdata[i],2))))
        #dist = class_acc(x.flatten(),trdata[i].flatten())
        
        if(dist < bestDistance or bestDistance == -1):
            bestDistance = dist
            baIndex = i

        if(bestDistance == 0):
            break


    return trlabels[baIndex]

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

datadict = unpickle('cifar-10-batches-py/data_batch_1')
datadictTest = unpickle('cifar-10-batches-py/test_batch')

X = datadict["data"]
Y = datadict["labels"]
X_Test = datadictTest["data"]
Y_Test = datadictTest["labels"]

#print(X.shape)

labeldict = unpickle('cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]

X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
X_Test = X_Test.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
Y = np.array(Y)
Y_Test = np.array(Y_Test)


newY_Test = []

for i in range(0,len(X_Test)):
    label = cifar10_classifier_1nn(X_Test[i],X,Y)
    newY_Test.append(label)

newY_Test = np.array(newY_Test)

print(class_acc(newY_Test,Y_Test))

#for i in range(X.shape[0]):
    # Show some images randomly
    #if random.random() > 0.999:
        #plt.figure(1)
        #plt.clf()
        #plt.imshow(X[i])
        #plt.title(f"Image {i} label={label_names[Y[i]]} (num {Y[i]})")
        #plt.pause(1)
        
