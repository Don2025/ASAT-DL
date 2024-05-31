from torch import nn

class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.num_directions = 2  # 双向
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, 
                          batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.num_directions * hidden_size, 2)  # 添加全连接层
        
    def forward(self, x):
        # x的大小为(batch_size, sequence_length, input_size)
        output, hidden = self.gru(x)
        # output的大小为(batch_size, sequence_length, num_directions * hidden_size)
        # hidden的大小为(num_layers * num_directions, batch_size, hidden_size)
        output = self.fc(output[:, -1, :])  # 取最后一个时刻的输出
        # output的大小为(batch_size, 2)
        return output

rnn_model = BiRNN(input_size=768, hidden_size=128, num_layers=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn_model.parameters(), lr=0.001)
num_epochs = 10
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.squeeze(1)  # 调整输入大小
        outputs = rnn_model(inputs)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

rnn_model.eval()

with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        inputs = inputs.squeeze(1)  # 调整输入大小
        outputs = rnn_model(inputs)  # 前向传播
        _, predicted = torch.max(outputs.data, 1)  # 预测类别
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy: {} %'.format(100 * correct / total))
