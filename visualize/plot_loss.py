import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


df = pd.read_csv('visualize/loss.csv', index_col=0)

df = df[['GPR - train_loss/total_loss', 'APPNP - train_loss/total_loss', '2HopConv - train_loss/total_loss', 'Base-SwinT - train_loss/total_loss']]

df.columns = ['GPR', 'APPNP', '2HopGCN', '1HopGCN(Base)']

y = df['GPR'].to_numpy()

yhat = savgol_filter(y, 51, 3) # window size 51, polynomial order 3

x = df.index.to_numpy()

plt.plot(x, yhat)
plt.savefig('./visualize/tmp.png')

print(x, y)

