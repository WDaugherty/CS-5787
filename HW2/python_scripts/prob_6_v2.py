import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sys


class Softmax:
    def __init__(self, epochs, learning_rate, batch_size, alpha, momentum):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.alpha = alpha
        self.momentum = momentum
        self.vel = None
        self.weight = None

    # Softmax function
    def softmax(input_batch, weights):
        exponent = np.exp(input_batch @ weights)
        exp_z = exponent / np.sum(exponent, axis=1, keepdims=True) 
        return exp_z 

    # Cross-entropy loss function
    def crossEntropyLoss(y_value_oneHot, y_pred):
        log_likelihood = np.log(np.sum(y_value_oneHot * y_pred, axis=1))
        loss = - np.mean(log_likelihood)
        return loss


    def predict_softmax_classifier(X_test, Y_test, weights):
        numberClasses = len(np.unique(Y_test))
        correct_predictions = np.zeros(numberClasses)
        total_class = np.zeros(numberClasses)
        
        y_pred = X_test @ weights
        predicted_classes = np.argmax(y_pred, axis=1)+1

        index = 0
        for c in np.unique(Y_test):
            correct_predictions[index] += np.sum((predicted_classes == c) & (Y_test == c))
            total_class[index] += np.sum(Y_test == c)
            index +=1

        mean_accuracy = np.mean(correct_predictions / total_class)    
        print("Test accuracy = {:.2f}".format(mean_accuracy))

        return predicted_classes, mean_accuracy



    def softmaxClassifier(X_train, Y_train, X_test, Y_test, form_weight_decay = "L2", learning_rate = 0.005, epochs = 100, batch_size = 10, weight_decay_factor=0.001, momentum = 0.8, mu= 0, sigma = 1.0):
        momentum_velocity = 0
        numberClasses = len(np.unique(Y_train))
        loss_values = []
        accuracy_values = []
        test_accuracy_values = []
        test_loss_values = []

        #Initializae weights from a Gaussian distribution. 
        weights = np.random.normal(loc = mu, scale = sigma,  size = (X_train.shape[1], numberClasses))

        #Loop for epochs (forward and backward passes)
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct_predictions = np.zeros(numberClasses)
            total_class = np.zeros(numberClasses)

            for batch_index in range(0,X_train.shape[0], batch_size):
                if (X_train.shape[0] - batch_index) > batch_size:
                    x_train_batch = X_train[batch_index: batch_index + batch_size]
                    y_train_batch = Y_train[batch_index: batch_index + batch_size]
                else:
                    x_train_batch = X_train[batch_index:]
                    y_train_batch = Y_train[batch_index:]

                #One Hot Encoding of output
                y_value_oneHot_batch = np.eye(numberClasses)[y_train_batch.astype(int)-1]
                y_pred = softmax(x_train_batch, weights)
                pred_error = y_pred - y_value_oneHot_batch 
                average_gradient = (np.transpose(x_train_batch) @ pred_error) / len(y_train_batch)

                # L2 or L1 Regularization 
                if form_weight_decay == "L2":
                    average_gradient = average_gradient + weight_decay_factor * 2*weights
                elif form_weight_decay == "L1":
                    average_gradient = average_gradient + weight_decay_factor * np.sign(weights)
                
                #Include momentum
                momentum_velocity = momentum_velocity * momentum - learning_rate * average_gradient
                
                #Update weights
                weights = weights + momentum_velocity

                # Calculate the loss for the batch
                batch_loss = crossEntropyLoss(y_value_oneHot_batch, y_pred) + 0.5 * weight_decay_factor * np.sum(weights**2)
                epoch_loss = epoch_loss + batch_loss

                # Calculate the accuracy for the batch
                predicted_classes = np.argmax(y_pred, axis=1)+1
                index = 0
                for c in np.unique(Y_train):
                    correct_predictions[index] += np.sum((predicted_classes == c) & (y_train_batch == c))
                    total_class[index] += np.sum(y_train_batch == c)
                    index +=1
            
                
            # Calculate the mean per-class loss and accuracy for each epoch
            mean_loss = epoch_loss / len(Y_train)
            mean_accuracy = np.mean(correct_predictions / total_class)
            
            # Save values for mean loss and accuracy
            loss_values.append(mean_loss)
            accuracy_values.append(mean_accuracy)
            
            #For Training Data
            print("Epoch {}: training accuracy ={:.2f}, loss ={:.4f}".format(epoch+1, mean_accuracy, mean_loss))

            #For Test Data
            predicted_classes, test_accuracy = predict_softmax_classifier(X_test, Y_test, weights)
            test_accuracy_values.append(test_accuracy)

            y_pred_test = softmax(X_test, weights)
            y_value_oneHot = np.eye(numberClasses)[Y_test.astype(int)-1]
            test_loss_values.append(crossEntropyLoss(y_value_oneHot, y_pred_test) + 0.5 * weight_decay_factor * np.sum(weights**2))
            


        return weights, loss_values, accuracy_values, test_accuracy_values, test_loss_values, predicted_classes