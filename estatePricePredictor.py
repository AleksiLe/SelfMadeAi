import numpy as np
import matplotlib.pyplot as plt

#real-estate.csv is from https://www.kaggle.com/quantbruce/real-estate-price-prediction

def formTrainingMatrixAndTestMatrix(fileName, folds): 
    #Training matrix rows that are later transposed to columns
    x1,x2,x3,x4,x5,y = [],[],[],[],[],[]
    #Test matrix rows that are later transposed to columns
    testX1,testX2,testX3,testX4,testX5,testY = [],[],[],[],[],[]
    #Counter for the folds
    counter = 1
    
    file = open(fileName, "r")
    #First line is the column names
    FirstColumn = file.readline()
    line = file.readline()
    while(len(line) > 0):
        row = line.replace("\n", "")
        id, transactionDate, houseAge, distanceToNearMrtStation,numberOfStores,latitude,longitude,housePriceOfUnitArea = row.split(",")
        #Using modulo and counter we split the data to training and test matrixes
        if(counter % folds == 0):
            testX1.append(1)
            testX2.append(float(transactionDate))
            testX3.append(float(houseAge))
            testX4.append(float(distanceToNearMrtStation))
            testX5.append(int(numberOfStores))
            testY.append(float(housePriceOfUnitArea))
        else:
            #we add 1 to the first column so that we can calculate the constant
            x1.append(1)
            #we only add those variables that we are interested to lists and after reading the file we form a matrix from them
            x2.append(float(transactionDate))
            x3.append(float(houseAge))
            x4.append(float(distanceToNearMrtStation))
            x5.append(int(numberOfStores))
            y.append(float(housePriceOfUnitArea))
        line = file.readline()
        counter += 1
    file.close()
    #form matrix from the lists
    trainingMatrix = np.array([x1,x2,x3,x4,x5,y])
    testMatrix = np.array([testX1,testX2,testX3,testX4,testX5,testY])
    #transpose the matrix so that the rows are the variables and columns are the observations
    trainingMatrix = trainingMatrix.transpose()
    testMatrix = testMatrix.transpose()
    #print(array)
    return trainingMatrix, testMatrix

def splitMatrixToXandY(matrix):
    Xmatrix = np.delete(matrix, [5], 1)
    Ymatrix = np.delete(matrix, [0,1,2,3,4], 1)
    return Xmatrix, Ymatrix

def findRegressionCoefficients(matrix):
    Xmatrix, Ymatrix = splitMatrixToXandY(matrix)
    XtX = np.dot(Xmatrix.transpose(), Xmatrix)
    XtY = np.dot(Xmatrix.transpose(), Ymatrix)
    betaMatrix = np.linalg.solve(XtX, XtY) #solve the equation XtX * betaMatrix = XtY

    return betaMatrix

def testModel(testMatrix, betaMatrix):
    Xmatrix, Ymatrix = splitMatrixToXandY(testMatrix)
    predictionMatrix = np.dot(Xmatrix, betaMatrix)
    errorSum = 0
    errorValues = []
    for i in range(0, len(predictionMatrix)):
        #print("Predicted value:",predictionMatrix[i][0],", Actual value:", Ymatrix[i][0])
        error = predictionMatrix[i][0] / Ymatrix[i][0]
        if (error < 1):
            error =  1 - error
        else:
            error = error - 1
        errorSum += error
        errorValues.append(Ymatrix[i][0] - predictionMatrix[i][0])
        #print("Error:", error)
    print("Average error:", errorSum / len(predictionMatrix))
    plt.scatter(errorValues,([0] * len(errorValues)))
    plt.grid()
    plt.xlabel("Error value")
    plt.ylabel("0")
    plt.show()
    return None


def main():
    #form scatter matrix for preview from formScatterMatrix()
    #formScatterMatrix()
    #Variables 
        #how many n-fold cross-validation are made,
        #input equals folds, 2 is 1 fold, 3 is 2 folds and so on
    folds = 3
        #File name where the data is
    fileName = "real-estate.csv"
    
    #form training matrix and test matrix with only the columns we need
    trainingMatrix, testMatrix = formTrainingMatrixAndTestMatrix(fileName, folds)
    #find regression coefficients
    betaMatrix = findRegressionCoefficients(trainingMatrix)
    #test the model
    testModel(testMatrix, betaMatrix)
    return None

main()