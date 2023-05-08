##基于决策树的手写数字分类
from sklearn import tree
import numpy as np
from sklearn.datasets import load_digits


dataset = np.load('D:\\VScode WorkStation\\CODE\\PythonPractice\\AI_Practice\\MNIST\\data\\MNIST\\raw.npz')
x_train = dataset['x_train']#所有自变量，用于训练的自变量。
y_train = dataset['y_train']#这是训练数据的类别标签。
x_test = dataset['x_test']#这是剩下的数据部分，用来测试训练好的决策树的模型。
y_test = dataset['y_test']#这是测试数据的类别标签，用于区别实际类型和预测类型的准确性。

classifier = tree.DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=21, min_samples_split=3,random_state=40,)
#classifier = tree.DecisionTreeClassifier(criterion='entropy',splitter='random',max_depth=None,min_samples_split=3,min_samples_leaf=2,min_weight_fraction_leaf=0.0,max_features=None,random_state=None,max_leaf_nodes=None,min_impurity_decrease=0.0,min_impurity_split=None,class_weight=None,)

x_train = x_train.reshape(60000, 784)#第一个数字是用来索引图片，第二个数字是索引每张图片的像素点
x_test = x_test.reshape(10000, 784)

classifier.fit(x_train, y_train)
score = classifier.score(x_test, y_test)
print(score)
#tree.plot_tree(classifier)
#import matplotlib.pyplot as plt
#plt.show()
#后三行是画出决策树
