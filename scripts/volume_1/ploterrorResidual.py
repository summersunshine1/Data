import pandas as pd
import matplotlib.pylab as plt

residual_path = "F:/kdd/dataSets/training/error_residual.csv"
residual = pd.read_csv(residual_path, encoding='utf-8')

ids = [1,2,3]
directions = [0,1]

def plot(id, direction):
    data = residual['residual'][(residual['id']==id)&(residual['direction'] == direction)]
    plt.plot(data)
    plt.show()

if __name__ == "__main__":
    for id in ids:
        for direct in directions:
            if id == 2 and direct == 1:
                continue
            plot(id, direct)
        