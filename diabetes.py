import torch
import torch.nn as nn
import pandas as pd
# 加载数据
data = pd.read_csv('diabetes2.csv')
# 分割训练集和测试集
train_data = data.sample(frac=0.99, random_state=42)
test_data = data.drop(train_data.index)
print(test_data)

#使用PyTorch来搭建LSTM-CNN模型
class LSTM_CNN(nn.Module):
    def __init__(self):
        super(LSTM_CNN, self).__init__()
        self.lstm = nn.LSTM(input_size=8, hidden_size=64, num_layers=2, batch_first=True)
        self.conv1 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.fc = nn.Linear(128, 4)

    def forward(self, x):
        x,_ = self.lstm(x)
        x = x.transpose(1, 0)
        x = self.conv1(x)
        x = x.transpose(1, 0)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 模型训练
model = LSTM_CNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    inputs = torch.tensor(train_data.drop('Outcome', axis=1).values).float()
    labels = torch.tensor(train_data['Outcome'].values).long()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('Epoch [%d/100], Loss: %.4f' %(epoch+1, loss.item()))

# 模型预测
inputs = torch.tensor(test_data.drop('Outcome', axis=1).values).float()
outputs = model(inputs)
_, predicted = torch.max(outputs.data, 1)
print('Predicted: ', predicted)



