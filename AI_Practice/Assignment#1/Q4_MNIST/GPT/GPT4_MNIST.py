import torch
import torchvision
import torch.utils.data
from torchvision import transforms

# 定义数据转换
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 加载训练集
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2
)

# 加载测试集
test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=32, 
    shuffle=False, 
    num_workers=2
)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 转换数据
X_train = train_dataset.data.numpy()
y_train = train_dataset.targets.numpy()
X_test = test_dataset.data.numpy()
y_test = test_dataset.targets.numpy()

# 重塑数据以适应 Scikit-learn 模型
X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

# 训练模型
models = [
    ('Logistic Regression', LogisticRegression(max_iter=10000)),
    ('Naive Bayes', GaussianNB()),
    ('SVM', SVC(kernel='linear')),
    ('Decision Tree', DecisionTreeClassifier())
]

for name, model in models:
    model.fit(X_train, y_train)
    
    # 验证模型性能
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.2%}")