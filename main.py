import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("D:\\python data\\xxianxinghuigui\\train_.csv")
data=data.dropna()
data.columns = ['id', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9','x10','x11', 'x12', 'x13', 'y']
print('打印数据看看')
print(data)
print('------------------------------------------------------------------')
#统计有多少行
y=np.stack(data['y'])
num = len(y)
x0 = np.ones(num)

#对每个元素进行标准化
x1=data['x1']
x2=data['x2']
x3=data['x3']
x4=data['x4']
x5=data['x5']
x6=data['x6']
x7=data['x7']
x8=data['x8']
x9=data['x9']
x10=data['x10']
x11=data['x11']
x12=data['x12']
x13=data['x13']

def standardzie(x):
  mu = x.mean()
  sigma = x.std()
  return (x-mu)/sigma

z1=standardzie(x1)
z2=standardzie(x2)
z3=standardzie(x3)
z4=standardzie(x4)
z5=standardzie(x5)
z6=standardzie(x6)
z7=standardzie(x7)
z8=standardzie(x8)
z9=standardzie(x9)
z10=standardzie(x10)
z11=standardzie(x11)
z12=standardzie(x12)
z13=standardzie(x13)


#完成x1到x13的 巨型多元数组
big_x=np.vstack((x0,z1,z2,z3,z4,z5,z6,z7,z8,z10,z11,z12,z13)).T
Big_x=np.stack(big_x)
print('打印出X矩阵')
print(Big_x)
print('-------------------------------------')
print('打印y向量')
print(y)
print('-------------------------------------')
#随机theta的数组
theta=np.random.rand(13)
print('打印出theta向量')
print(theta)
print('------------------------------------------------')

def f(x):
  return np.dot(x,theta)

print('打印出f（X）的矩阵看看')
print(f(Big_x))
print('------------------------------------------------')
def E(x,y):
  return 0.5*np.sum((y-f(x))**2)

erro=E(Big_x,y)

print('打印误差看看')
print(erro)
print('------------------------------------------------')
print('打印学习过程的向量相减')
print(f(Big_x)-y)
print('------------------------------------------------')
#开始循环
ETA=1e-4

diff =1

LAMBTA=10

while diff>1e-6:
  #正则化项目，0不能用正则化
  reg_term=LAMBTA*np.hstack([0,theta[1:]])
  theta=theta-ETA*(np.dot(f(Big_x)-y,Big_x)+reg_term)
  current_error=E(Big_x,y)
  diff=erro-current_error
  erro=current_error
  #输出日记
  log='误差为{}'
  print(log.format(erro))

#开始输出结果
print('打印输出原结果')
output_y=f(Big_x)
print(output_y)
print('------------------------------------------------')
#首先导入test.csv
test=pd.read_csv("D:\\python data\\xxianxinghuigui\\test_.csv")

x0=test['0']
x1=test['1']
x2=test['2']
x3=test['3']
x4=test['4']
x5=test['5']
x6=test['6']
x7=test['7']
x8=test['8']
x9=test['9']
x10=test['10']
x11=test['11']
x12=test['12']

#统计有多少行
xxx=np.stack(test['0'])
num = len(xxx)
xhang = np.ones(num)

test_x=np.vstack((xhang,x0,x1,x2,x3,x4,x5,x6,x7,x8,x10,x11,x12,)).T
print('打印test_x的矩阵')
print(test_x)
print('------------------------------------------------')

#开始激情输出
outcome=f(test_x)
print(outcome)
outcome=np.vstack(outcome)
print(outcome)
np.savetxt("D:\\python data\\result.xlsx", outcome)





#补充前面的导入和标准化的繁琐：后续又写了一次，简化的：
# data_region = pd.read_csv("D:\\python data\\xxianxinghuigui\\train_.csv")
# data_solved = data_region.dropna()
# print(data_region[:8])
# print(data_solved[:8])
# data_narray01 = np.array(data_solved)
# data_narray = data_narray01[:, 1:-1]
# # 转化成浮点型：
# data_narray = data_narray.astype(float)
# print(data_narray[:8])
#
# print('over')
#
# # 先T后开始标准化
# data_narrayT = data_narray.T
# print(data_narrayT)
# #开始标准化：
# for i in range(len(data_narrayT)):
#   pinjun = np.mean(data_narrayT[i])
#   fangcha = np.std(data_narrayT[i])
#   data_narrayT[i] = (data_narrayT[i] - pinjun) / fangcha
# print(data_narrayT)
# data_narray = data_narrayT.T
# print(data_narray)
# print('标准化完成,最后的数组为data_narray')
#
# # 开始弄出方程啊a,b,c。。。的数组
# theta = np.random.randn(len(data_narrayT))
# print(theta)
# print('数组打印完成')