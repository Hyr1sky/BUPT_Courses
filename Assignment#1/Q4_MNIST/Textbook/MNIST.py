"""from sklearn.datasets import load_digits
from matplotlib import pyplot as plt  # 导入绘图模块

digit = load_digits()  # 加载手写数字数据集
plt.subplots(5,10)  # 5行10列子图

for i, data in enumerate(zip(digit.data[:50], digit.target[:50])):  # zip函数将特征和标签打包成二元组列表
        plt.subplot(5, 10, i + 1, title=str(data[1]))  # 设置子图标题为标签
        plt.xticks([])  # 不显示横坐标
        plt.yticks([])  # 不显示总坐标
        plt.imshow(data[0].reshape((8, 8)), cmap=plt.cm.gray_r)  # 灰度图形显示
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

#-----数据集加载-----#

# 规定训练数据集与测试数据集
training_data = datasets.MNIST(
    root="./data",  # 存储训练数据集的路径
    train=True,  # 指定训练数据集
    download=True,  # 如果root路径不可用，则从Internet下载数据
    transform=ToTensor(),  # 指定特征和标签转换为张量
)
test_data = datasets.MNIST(
    root="./data",  # 存储测试数据集的路径
    train=False,  # 指定测试数据集
    download=True,
    transform=ToTensor(),
)

batch_size = 64  # 设置一次输入网络的样本数量为64
# 实例化训练数据加载器
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
# 实例化测试数据加载器
test_dataloader = DataLoader(test_data, batch_size=batch_size)

#-----数据集加载-----#

# 定义CNN网络
class CNN(nn.Module):  # 从nn.Module派生CNN类
    def __init__(self):  # 定义初始化函数
        super(CNN, self).__init__()  # 调用基类初始化函数
        # 定义第一个卷积层，实际输入张量shape应为(N, 1, 28, 28)，N为batch_size大小
        self.conv1 = nn.Sequential(
            nn.Conv2d(  # 定义卷积核
                in_channels=1,  # 输入特征图的通道数
                out_channels=16,  # 输出特征图的通道数
                kernel_size=5,  # 卷积核大小为5X5
                stride=1,  # 滑动步长
                padding=2,  # 边界扩展填充
            ),  # 不难推断输出张量shape为(N, 16, 28, 28)
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(kernel_size=2),  # 最大池化，输出特征图为(N, 16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # 定义第二个卷积层
            nn.Conv2d(16, 32, 5, 1, 2),  # 定义卷积核，输出特征图为(N, 32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),  # 最大池化，输出特征图变为(N, 32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)  # 定义全连接层，输出为10个类别

    def forward(self, x):  # 定义前向传播函数
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 将特征图平铺，shape由(N, 32, 7, 7)变为(N, 1568)
        output = self.out(x)  # 输出为(N, 10) 全连接层输出十维张量
        return output  # 返回最后输出


# 加载模型到cpu或gpu中
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learn_rate = 1e-3  # 定义学习率
model = CNN().to(device)  # 实例化模型，并加载到cpu或gpu中
loss_fn = nn.CrossEntropyLoss()  # 定义交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)  # 设置优化器

# 训练模型

def mnist_train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)  # 得到数据集的大小
    model.train()  # 模型进入训练状态，权值会进行学习
    for batch, (X, y) in enumerate(dataloader):  # 每个循环取batch_size个样本
        X, y = X.to(device), y.to(device)  # 样本特征和标签加载到计算环境
        pred = model(X)  # 得到预测结果,shape为(64, 10)
        loss = loss_fn(pred, y)  # 计算损失
        optimizer.zero_grad()  # 将参数梯度全部设置为0
        loss.backward()  # 通过损失计算所有参数梯度
        optimizer.step()  # 通过计算得到的参数梯度对网络参数进行更新

        if batch % 100 == 0:
            loss = loss.item()  # item()方法得到张量里的元素值，丢弃梯度信息
            current = batch * len(X)  # 得到已训练的样本数量
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

# 测试模型

def mnist_test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()  # 模型进入测试状态，权值会被固定住，不会被改变
    test_loss, correct = 0, 0
    with torch.no_grad():  # 该上下文管理器中的数据不会跟踪梯度，加快计算过程
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()  # 累计损失
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # 得到正确样本数量
    test_loss /= num_batches
    correct /= size
    print(f"Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__=='__main__':
    #  正式开始训练和验证
    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t + 1}\n --------")
        mnist_train(train_dataloader, model, loss_fn, optimizer)
        mnist_test(test_dataloader, model, loss_fn)
    print("Done")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
