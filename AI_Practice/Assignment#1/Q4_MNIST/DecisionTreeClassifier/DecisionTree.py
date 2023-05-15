import numpy as np
from sklearn.datasets import load_digits
from sklearn import tree
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    mnist = load_digits()
    x,test_x,y,test_y = train_test_split(mnist.data, mnist.target, test_size = 0.25, random_state = 40)
    
    model = tree.DecisionTreeClassifier()
    model.fit(x, y)
    acc = model.score(test_x, test_y)
    print("The accuracy is %.3f" % acc)
    z = model.predict(test_x)

    print('准确率:',np.sum(z == test_y)/z.size)
