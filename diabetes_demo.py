import torch
import matplotlib.pyplot as plt
import numpy as np
# 注意这里必须写成两维的矩阵
#文件名，以‘，’作为分割符，常用32位浮点数
xy = np.loadtxt('diabetes2.csv', delimiter=',', dtype=np.float32,skiprows=1)
x_data = torch.from_numpy(xy[:, :-1]) #特征信息
y_data = torch.from_numpy(xy[:, [-1]]) #目标分类

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 线性模型: y = w*x + b
        # 在线性模型 Linear 类中,第一次训练时的参数 w 和 b 都是给的随机数
        self.linear1 = torch.nn.Linear(8, 6) #输入数据x的特征是8维，x有8个特征
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    # __call__() 中会调用这个函数！
    # 类里面定义的每个函数都需要有个参数self,来代表自己，用来调用类中的成员变量和方法
    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

# model为可调用的！ 实现了 __call__()
model = Model()
print(model)
# 构建损失函数和优化器:BCELoss---运用交叉熵计算两个分布之间的差异
criterion = torch.nn.BCELoss(reduction='mean')  # 二分类交叉熵损失函数
# -- 指定优化器（其实就是有关梯度下降的算法，负责），这里将优化器和model进行了关联
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epoch_list = []
loss_list = []

for epoch in range(5000):
    y_pred = model(x_data)  # 直接把整个数据都放入了
    # 计算训练输出的值和真实的值之前的分布差异
    loss = criterion(y_pred, y_data)

    print("epoch=",epoch, "Loss=",loss.item())
    epoch_list.append(epoch)
    loss_list.append(loss.item())
    #重置梯度,梯度清零
    optimizer.zero_grad()  # 会自动找到所有的w和b进行清零！优化器的作用 （为啥这个放到loss.backward()后面清零就不行了呢？）
    # 计算梯度反向传播
    loss.backward()
    #优化器根据梯度值进行优化，更新梯度
    optimizer.step()  # 会自动找到所有的w和b进行更新，优化器的作用！

#训练过程可视化
plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

#  测试
xy = np.loadtxt('diabetes_test.csv', delimiter=',', dtype=np.float32,skiprows=1)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, -1])

x_test = torch.Tensor(x_data)
y_test = model(x_test)  # 预测

# 对比预测结果和真实结果
for index, i in enumerate(y_test.data.numpy()):
    if i[0] > 0.5:
        print(1, int(y_data[index].item()))
    else:
        print(0, int(y_data[index].item()))

print("11111111111111111111")
