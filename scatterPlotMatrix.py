#real-estate.csv is from https://www.kaggle.com/quantbruce/real-estate-price-prediction
#this file was used to form the scatter plot matrix
import pandas as pd
import matplotlib.pyplot as plt

""" 
print(data.columns)
Index(['No', 'transaction date', 'house age',
       'distance to the nearest MRT station', 'number of convenience stores',
       'latitude', 'longitude', 'house price of unit area'],
      dtype='object') 
"""

def readCsv(path):
    data=pd.read_csv(path, sep=",")
    return pd.DataFrame(data)  

def formScatterMatrix():
    data = readCsv("real-estate.csv")
    #Calculate pearson values. 
    print(data.corr(method='pearson'))

    #first column

    plt.scatter(data[data.columns[1]], data[data.columns[2]])
    plt.grid()
    plt.xlabel(data.columns[1])
    plt.ylabel(data.columns[2])
    plt.show()
    
    plt.scatter(data[data.columns[1]], data[data.columns[3]])
    plt.grid()
    plt.xlabel(data.columns[1])
    plt.ylabel(data.columns[3])
    plt.show()

    plt.scatter(data[data.columns[1]], data[data.columns[4]])
    plt.grid()
    plt.xlabel(data.columns[1])
    plt.ylabel(data.columns[4])
    plt.show()

    plt.scatter(data[data.columns[1]], data[data.columns[7]])
    plt.grid()
    plt.xlabel(data.columns[1])
    plt.ylabel(data.columns[7])
    plt.show()

    #second column

    plt.scatter(data[data.columns[2]], data[data.columns[3]])
    plt.grid()
    plt.xlabel(data.columns[2])
    plt.ylabel(data.columns[3])
    plt.show()

    plt.scatter(data[data.columns[2]], data[data.columns[4]])
    plt.grid()
    plt.xlabel(data.columns[2])
    plt.ylabel(data.columns[4])
    plt.show()

    plt.scatter(data[data.columns[2]], data[data.columns[7]])
    plt.grid()
    plt.xlabel(data.columns[2])
    plt.ylabel(data.columns[7])
    plt.show()

    #third column

    plt.scatter(data[data.columns[3]], data[data.columns[4]])
    plt.grid()
    plt.xlabel(data.columns[3])
    plt.ylabel(data.columns[4])
    plt.show()

    plt.scatter(data[data.columns[3]], data[data.columns[7]])
    plt.grid()
    plt.xlabel(data.columns[3])
    plt.ylabel(data.columns[7])
    plt.show()

    #last column

    plt.scatter(data[data.columns[4]], data[data.columns[7]])
    plt.grid()
    plt.xlabel(data.columns[4])
    plt.ylabel(data.columns[7])
    plt.show()
    return None