import pandas as pd
import numpy as np
import matplotlib.pylab as plt

weather_path = "F:/kdd/dataSets/training/weather (table 7)_training.csv"

def readData(path):
    data = pd.read_csv(path,encoding='utf-8')
    df = data
    print(data.describe())
    print(data['pressure'].unique())
    plt.hist(df['wind_direction'], bins = 10, range =(df['wind_direction'].min(),df['wind_direction'].max()))
    plt.show()
    
if __name__ == "__main__":
    readData(weather_path)