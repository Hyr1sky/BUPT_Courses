import numpy as np
from sklearn.datasets import load_digits
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    mnist = load_digits()
    x,test_x,y,test_y = train_test_split(mnist.data, mnist.target, test_size = 0.25, random_state = 40)
    
    model1 = GaussianNB()
    model1.fit(x, y)
    acc1 = model1.score(test_x, test_y)
    print("GuassianNB accuracy is %.3f" % acc1)
    z1 = model1.predict(test_x)
    print('准确率:',np.sum(z1 == test_y)/z1.size)

    model2 = MultinomialNB()
    model2.fit(x, y)
    acc2 = model2.score(test_x, test_y)
    print("MultionmialNB accuracy is %.3f" % acc2)
    z2 = model2.predict(test_x)
    print('准确率:',np.sum(z2 == test_y)/z2.size)

    model3 = BernoulliNB()
    model3.fit(x, y)
    acc3 = model3.score(test_x, test_y)
    print("BernoulliNB accuracy is %.3f" % acc3)
    z3 = model3.predict(test_x)
    print('准确率:',np.sum(z3 == test_y)/z3.size)
