import itertools
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from Pr6 import Softmax
from datasets import load_dataset


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='iso-8859-1')
    return dict

def loadData(file):
    dict = unpickle(file)
    X = []
    y = []

    for k in range(dict["data"].shape[0]):
        label = dict["labels"][k]
        srcImg = dict["data"][k]
        X.append(srcImg)
        y.append(label)
    X = np.array(X)
    y = np.array(y)
    return X, y

def createFig(trainLosses):
    plt.plot(trainLosses, label="Train loss")
    plt.legend(loc='best')
    plt.title("Train Loss varying with Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

def createFig_Acc(trainAcc, testAcc):

    plt.plot(trainAcc, label="Train Accuracy")
    plt.plot(testAcc, label="Test Accuracy")
    plt.legend(loc='best')
    plt.title("Mean per class Accuracy varying with Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Mean per class Accuracy")
    plt.show()

def getConfusionMatrix(actualLabel, predictedLabel, numOfClass):
    confMtrx =[]
    for _ in range(numOfClass):
        confMtrx.append([])
        for _ in range(numOfClass):
            confMtrx[-1].append(0)

    for sampleNum in range(actualLabel.shape[0]):
        confMtrx[int(actualLabel[sampleNum])][int(predictedLabel[sampleNum])] += 1
    confMtrx = np.array(confMtrx)
    return confMtrx

def plotConfusionMatrix(s81, xTest, actualLabel, classes, normalize=False,
                        title='Confusion matrix', cmap=plt.cm.Blues):
    predY = s81.predict(xTest)
    predY = predY.reshape((-1, 1))
    confMtrx = getConfusionMatrix(actualLabel, predY, 10)
    if normalize:
        confMtrx = confMtrx.astype('float') / confMtrx.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix without normalization')

    print(confMtrx)

    plt.imshow(confMtrx, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = confMtrx.max() / 2.
    for i, j in itertools.product(range(confMtrx.shape[0]), range(confMtrx.shape[1])):
        plt.text(j, i, format(confMtrx[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confMtrx[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == '__main__':
    #Reading data into np.arrays as vectors
    dataset = load_dataset('cifar10')

    X_train = (dataset['train'][:]['img'])
    Y_train = np.array(dataset['train'][:]['label'])
    X_test = (dataset['test'][:]['img'])
    Y_test = np.array(dataset['test'][:]['label'])

    X_train = np.asarray([np.asarray(x) for x in X_train])
    X_test = np.asarray([np.asarray(x) for x in X_test])


    #Reshape as vectors
    X_train = X_train.reshape((X_train.shape[0], np.prod(X_train.shape[1:])))
    X_test = X_test.reshape((X_test.shape[0], np.prod(X_test.shape[1:])))

    # We normalize features so they are between -1 and 1 
    X_train = (X_train/255)*2-1
    X_test = (X_test/255)*2-1

    print(X_train.shape)
    print(X_test.shape)


    epochs = 100
    learningRate = 0.001
    batchSize = 20
    alpha = 0.0001
    momentum = 0.005

    sm = Softmax(epochs=epochs, learning_rate=learningRate, batch_size=batchSize,
                 alpha=alpha, momentum=momentum)
    
    trainLosses, testLosses, trainAcc, testAcc = sm.train(X_train, Y_train, X_test, Y_test)
    createFig(trainLosses)
    
    createFig_Acc(trainAcc, testAcc)

    plotConfusionMatrix(sm, X_test, Y_test, "0123456789",
                        normalize=True, title='Normalized confusion matrix')
    plt.show()