import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# 自定义函数 e指数形式
def func(x, a, b, c):
    return a * np.sqrt(x) * (b * np.square(x) + c)
def up1(x,a,b,c,d,e):
    return (a+b*x)*np.exp(c*x)+d*np.exp(e*x)

# 定义x、y散点坐标
x = [20, 30, 40, 50, 60, 70]
x = np.arange(0,22,2)
num = [32.41882,33.08941,35.13412,37.56235,40.37882,43.2,45.38824,47.79294,50.08,52.58588,54.8]
y = np.array(num)

# 非线性最小二乘法拟合
popt, pcov = curve_fit(up1, x, y)
# 获取popt里面是拟合系数
print(popt)
a = popt[0]
b = popt[1]
c = popt[2]
d = popt[3]
e = popt[4]
yvals = up1(x, a, b, c,d,e)  # 拟合y值
r2=r2_score(y,yvals)
print('popt:', popt)
print('系数a:', a)
print('系数b:', b)
print('系数c:', c)
print('系数d:', d)
print('系数e:', e)
print('系数pcov:', pcov)
print('系数yvals:', yvals)
print('拟合度R2:',r2)
# 绘图
plot1 = plt.plot(x, y, 's', label='original values')
plot2 = plt.plot(x, yvals, 'r', label='polyfit values')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=4)  # 指定legend的位置右下角
plt.title('curve_fit')
plt.show()


