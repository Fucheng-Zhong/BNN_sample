'''
pytorch 基本入门操作
from
https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
'''
import torch
from torch import nn
from torch.utils.data import DataLoader # DataLoader 用于讲训练/测试数据打包 划分batch/打乱顺序等 以方便多线程计算
from torchvision import datasets  # 一些常用的数据库，例如手写数据等，用于训练和测试
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os

# 1. Working with data
# 下载图像数据集，包含10类图像，训练用。 Download training data from open datasets.
# 包含的数据集详见： https://pytorch.org/vision/stable/datasets.html
# 下载完不会再次下载
training_data = datasets.FashionMNIST(
    root="data", # 数据保存目录 自动创建
    train=True,
    download=True,
    transform=ToTensor(),
)
# 下载数据集，测试用。 Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data", # 数据保存目录 自动创建
    train=False,
    download=True,
    transform=ToTensor(),
)

# 用于显示部分数据
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# 自动批处理数据
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
batch_size = 64
# Create data loaders.
# supports automatic batching, sampling, shuffling and multiprocess data loading
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# 2. 创建网络模型
# 模型是继承于 nn.Module 的类
# 在 __init__ function 里定义层。
# 在 forward function 指定层的连接。
# https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self): # 定义层
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten() #平坦化 28*28 -> 一维
        self.linear_relu_stack = nn.Sequential( # 网络序列
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x): # 层的连接
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device) # 类实例化
print(model)

# 3. 参数优化 Optimizing the Model Parameters（训练）
# 优化网络参数需要 损失函数loss function 和优化器optimizer （SGD或Adam等）
# https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
loss_fn = nn.CrossEntropyLoss() #交叉熵
# 
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) #SGD随机梯度下降 优化器

if os.path.exists('./model.pth'): # 在已有模型上继续训练
    model.load_state_dict(torch.load("model.pth")) 

# 在一个训练周期内，模型被喂入训练数据（fed to it in batches），根据预测误差反向调节（backpropagates）模型参数
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()   # 设置模型为训练模式 改变网络参数
    for batch, (X, y) in enumerate(dataloader): # 分bathc喂入
        X, y = X.to(device), y.to(device)   # 张量迁移到device，这里并不是重点。如有GPU的话只是起到加速训练等作用

        # Compute prediction error
        pred = model(X) # 给出预测值
        loss = loss_fn(pred, y)# 损失函数是 预测值pred 和 标签值y 的函数，以衡量它们之间的差异

        # Backpropagation
        optimizer.zero_grad() # 清空本轮batch的训练梯度
        loss.backward() # 反向传播，计算梯度
        optimizer.step() # 更新网络参数

        if batch % 100 == 0: # 每经过100个batchs输出一次训练结果
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# 用于检查模型的表现 确保模型在学习 其实叫validation（评估集）才对
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() # 设置模型为评估模式 用于评估或预测 不改变网络参数
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# 训练
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")


# 4. 保存和装载模型 Saving && Loading Models
# A common way to save a model is to serialize the internal state dictionary (containing the model parameters).
torch.save(model.state_dict(), "model.pth") # 保存模型到 ./model.pth 中
print("Saved PyTorch Model State to model.pth")

#装载模型
# The process for loading a model includes re-creating the model structure and loading the state dictionary into it.
model = NeuralNetwork() # 实例化
model.load_state_dict(torch.load("model.pth")) 


# 5.预测
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval() # Sets the module in evaluation mode.
x, y = test_data[0][0], test_data[0][1] # 测试集的第一个样本
with torch.no_grad(): # 不进行计算图构建
    pred = model(x) # 预测值
    predicted, actual = classes[pred[0].argmax(0)], classes[y] # 预测值和真实值
    print(f'Predicted: "{predicted}", Actual: "{actual}"')