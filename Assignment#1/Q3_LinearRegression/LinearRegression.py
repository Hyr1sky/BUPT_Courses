import numpy as np
from sklearn. linear_model import LinearRegression
from sklearn. model_selection import train_test_split
import matplotlib. pyplot as plt

#设置参数
p = 1000 #采样点数
a = 0 #均值
b = 1 #方差
#在区间(0，100)中生成p个均匀采样点
x = np.linspace(0, 100, p)
#生成y
n = np.random.normal(a, b, p)
y = (x + 5)**2 + 10 + n
#将数据整形为线性回归所需格式
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
#将数据集分为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#使用线性回归拟合数据
model = LinearRegression()
model.fit(x_train,y_train)
#预测
y_pred = model.predict(x_test)
#输出线性方程
print(f"线性方程: y = {model.coef_[0][0]}*x + {model.intercept_[0]}")
#可视化结果
plt.scatter(x_test, y_test, color = 'blue', label = '实际数据')
plt.plot(x_test, y_pred, color = 'red', label = '预测结果')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()