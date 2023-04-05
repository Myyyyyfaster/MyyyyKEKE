import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#导入数据，并去除列中为0的列
data_region=pd.read_csv("D:\\python data\\softmax回归\\train.csv")
# print(data_region)
data_region=np.array(data_region)[:,1:]
# print(data_region)
data_region=pd.DataFrame(data_region)
# print(data_region)
data_region=data_region.replace(0,np.nan)
data_region=data_region.dropna(how='all',axis=1)
# print(data_region)
data_region=data_region.replace(np.nan,0)
# print(data_region)
data_region=np.array(data_region)
# print(data_region)
# print(len(data_region.T))

def to_x0(x):
    x0=np.ones([len(x),1])
    return np.hstack([x0,x])

train_y=data_region[:,0]
train_x=data_region[:,1:]
# print(len(train_x.T))

mu=np.mean(train_x,axis=0)
sigma=np.std(train_x,axis=0)
# print(len(mu))
# print(mu)
# print(len(sigma))
# print(sigma)
def standardize(x):
    return (x-mu)/sigma

train_z=standardize(train_x)
# print(train_z)
train_z=to_x0(train_z)
# print(train_z)
# print(len(train_z.T))

theta=np.random.rand(len(train_z.T))
# print(len(theta))

W=np.exp(-np.dot(train_z,theta.T))/np.sum(np.exp(-np.dot(train_z,theta.T)))
# print(W)
print(len(W))

W_01=np.where(W>=0.5,1,0)
# print(W_01)
# print(np.mean(W_01))
# print(np.max(W_01))

#梯度下降还没做，



print('over')