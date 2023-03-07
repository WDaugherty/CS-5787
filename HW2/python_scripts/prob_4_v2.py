import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random



class Regression:

    def __init__(self, epochs, learning_rate, batch_size, alpha, model):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.alpha = alpha
        self.model = model
        self.weight = None

    def train(self, x_train, y_train, x_test, y_test):
        dim = x_train.shape[1]
        self.weight = np.random.rand(dim)
        trainLosses = []
        testLosses = []
        trainMSE = []
        testMSE = []
        for epoch in range(self.epochs):
            trainLoss = self.SGD(x_train, y_train)
            if model == "L1":
                testLoss, _ = self.computeLossL1(x_test, y_test)
            elif model == "L2":
                testLoss, _ = self.computeLossL2(x_test, y_test)
            elif model == "poisson":
                testLoss, _ = self.poissonRegression(x_test, y_test)
            trainLosses.append(trainLoss)
            testLosses.append(testLoss)
            trainMSE.append(musicMSE(x_train @ self.weight, y_train))
            testMSE.append(musicMSE(x_test @ self.weight, y_test))
            print("{:d}\t->\tTrainL : {:.7f}\t|\tTestL : {:.7f}\t|\tTrainMSE : {:.7f}\t|\tTestMSE: {:.7f}"
                  .format(epoch, trainLoss, testLoss, trainMSE[-1], testMSE[-1]))
        return trainLosses, testLosses, trainMSE, testMSE

    def SGD(self, x, y):
        losses = []
        randInd = random.sample(range(x.shape[0]), x.shape[0])
        x = x[randInd]
        y = y[randInd]
        for i in range(0, x.shape[0], self.batch_size):
            Xbatch = x[i: i + self.batch_size]
            ybatch = y[i: i + self.batch_size]
            if model == "L1":
                loss, dw= self.computeLossL1(Xbatch, ybatch)
            elif model == "L2":
                loss, dw = self.computeLossL2(Xbatch, ybatch)
            elif model == "poisson":
                loss, dw = self.poissonRegression(Xbatch, ybatch)
            self.weight -= self.learning_rate * dw
            losses.append(loss)
        return np.sum(losses) / len(losses)

    def computeLossL1(self, x, y):
        samples = x.shape[0]
        y_star = np.dot(x, self.weight)
        loss = np.sum((y_star - y) ** 2)
        regLoss = self.alpha * np.linalg.norm(self.weight,ord=1)
        totalLoss = loss + regLoss

        grad = (-2) * np.dot(y-y_star, x) + self.alpha

        return totalLoss/samples, grad

    def computeLossL2(self, x, y):
        samples = x.shape[0]
        scores = np.dot(x, self.weight)
        y_star = scores
        loss = np.sum((y_star - y) ** 2)
        regLoss = self.alpha * np.sum(self.weight ** 2)
        totalLoss = loss + regLoss

        grad = 2 * np.dot(y_star-y, x) + 2*self.alpha*np.linalg.norm(self.weight,ord=1)
        return totalLoss / samples, grad

    def poissonRegression(self, x, y):
        samples = x.shape[0]
        y_star = np.dot(x, self.weight)
        loss = np.exp(y_star) - y * y_star
        totalLoss = np.sum(loss)
        grad = np.dot(x.T, np.exp(y_star) - y) / samples
        return totalLoss / samples, grad


def loadMusicData(fname, addBias=True):
    data = np.loadtxt(fname, delimiter=',')
    X1 = data[:800, 1:]
    y1 = data[:800, 0].astype(int)
    X2 = data[800:, 1:]
    y2 = data[800:, 0].astype(int)
    if addBias:
        y1 = y1 - 1922
        y2 = y2 - 1922
    return y1, X1, y2, X2

def musicMSE(pred, gt):
        return np.mean(np.square(np.subtract(gt,np.round(pred))))

def analyseData(trainY, testY):
    data = np.append(trainY, testY)
    plt.hist(data, bins = 90)
    plt.title("Histogram of the labels in the train and test set")
    plt.xlabel("Label of Year")
    plt.ylabel("Number")
    plt.show()

def createFig(trainLosses, testLosses, trainMSE, testMSE):
    plt.subplot(1, 2, 1)
    plt.plot(trainLosses, label="Train loss")
    plt.plot(testLosses, label="Test loss")
    plt.legend(loc='best')
    plt.title("Average Train Cross Entropy Loss")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Cross Entropy Loss")

    plt.subplot(1, 2, 2)
    plt.plot(trainMSE, label="Train MSE")
    plt.plot(testMSE, label="Test MSE")
    plt.legend(loc='best')
    plt.title("Mean MSE varying with Epochs")
    plt.yscale("log")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Mean MSE")

    plt.show()


def pseudoinverse(trainFeat, trainYears, testFeat, testYears):
    alpha=0.001
    # Finding weights using pseudoinverse (Bias is included as a Feature)
    weights_pseudoinverse = np.linalg.inv(np.transpose(trainFeat) @ trainFeat + alpha*np.identity(trainFeat.shape[1])) @ np.transpose(trainFeat) @ trainYears

    # Predicting using pseudoinverse
    y_predict_pseudoinverse = testFeat @ weights_pseudoinverse

    #Calculating MSE from pseudoinverse
    error_pseudoinverse = musicMSE(y_predict_pseudoinverse, testYears)
    
    
    return print(error_pseudoinverse)


if __name__ == '__main__':
    """
    Main method
    """
def loadMusicData(filename: str, add_bias: bool = True) -> tuple:
    """Loads music data from a file and returns training and testing data.

    Args:
        filename (str): The name of the file containing the music data.
        add_bias (bool, optional): Whether or not to add a bias term to the feature vectors. Defaults to True.

    Returns:
        tuple: A tuple containing the training years, training features, testing years, and testing features.
    """
    with open(filename, "r") as file:
        data = file.readlines()

    # Parse the data
    data_new = [row.strip().split(",") for row in data]
    
    # Split into training and testing data
    training_data = np.array(data_new[:463714])
    testing_data = np.array(data_new[463714:])
    
    # Split features and labels
    train_years, test_years = training_data[:,0].astype(int).reshape(-1,1),testing_data[:,0].astype(int).reshape(-1,1)
    train_features,test_features = training_data[:,1:].astype(float),testing_data[:,1:].astype(float)
  

    # Add the bias term to the features if requested
    if add_bias:
        train_features = np.append(np.ones([len(train_features),1]), train_features, 1)
        test_features = np.append(np.ones([len(test_features),1]), test_features, 1)

    return train_years, train_features, test_years, test_features

def musicMSE(predictions: np.ndarray, ground_truths: np.ndarray) -> float:
    """
    This function calculates the Mean Squared Error (MSE) between the predicted and ground truth music data.
    It takes two input numpy arrays, `predictions` and `ground_truths`, and returns the calculated MSE value.

    Args:
    - predictions: A numpy array of predicted music data
    - ground_truths: A numpy array of ground truth music data

    Returns:
    - A float value representing the calculated MSE
    """

    # Round the predicted music data to the nearest integer value
    rounded_predictions = np.rint(predictions)

    # Calculate the MSE between the rounded predictions and ground truth music data
    mse = np.square(rounded_predictions - ground_truths).mean()

    return mse

#     def loadMusicData(filename, addBias=True):
#         # reading data from the file
#         data = pd.read_csv(filename, delimiter=',')

#         data_years = data.iloc[:, 0].values
#         data_feat = data.iloc[:, 1:].values

#         # Normalize training and test sets using z-score normalization
#         feat_mean = np.mean(data_feat, axis=0)
#         feat_std = np.std(data_feat, axis=0)
#         norm_feat = (data_feat - feat_mean) / feat_std

#         train_data = data_feat[:463714]
#         test_data = data[463714:]

#         # split into target and features
#         split_idx = 463714
#         train_years = data_years[:split_idx]
#         train_feat = norm_feat[:split_idx]
#         test_years = data_years[split_idx:]
#         test_feat = norm_feat[split_idx:]

#         # append bias term
#         if addBias:
#             train_feat = np.hstack((train_feat, np.ones((train_feat.shape[0], 1))))
#             test_feat = np.hstack((test_feat, np.ones((test_feat.shape[0], 1))))

#         return train_years, train_feat, test_years, test_feat

#     def musicMSE(pred, true):
#         pred = np.round(pred)
#         mse = np.mean((pred - true) ** 2)
#         return mse

# trainYears, trainFeat, testYears, testFeat = loadMusicData("/Users/wdaugherty/Cornell_Tech_DL/CS-5787/HW2/data/YearPredictionMSD.txt")

trainYears, trainFeat, testYears, testFeat = loadMusicData("/Users/wdaugherty/Cornell_Tech_DL/CS-5787/HW2/data/YearPredictionMSD.txt",
add_bias = True)
print("---", len(trainYears), "----")



epochs = 100
learning_rate = 0.00000000001
batch_size = 32
alpha = 0.00001
model = "L1"
momentum = 0.005

pseudoinverse(trainFeat, trainYears, testFeat, testYears)


rg = Regression(epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, alpha=alpha, model=model)
trainLosses, testLosses, trainMSE, testMSE = rg.train(trainFeat, trainYears, testFeat, testYears)
createFig(trainLosses, testLosses, trainMSE, testMSE)

