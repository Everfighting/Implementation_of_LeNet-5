import torch 
import torch.nn as nn 
import torchvision 
import torchvision.transforms as transforms

# 设置超参数
batch_size = 64 
num_classes = 10 
learning_rate = 0.001
num_epochs = 10 

# 获取设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 获取本地下载好的文件
# 生成dataset
train_dataset = torchvision.datasets.MNIST(root = './data', train = True, transform = transforms.Compose([   
        transforms.Resize((32,32)),
        transforms.ToTensor(), 
        transforms.Normalize(mean = (0.1307,), std = (0.3081,))
        ]))


test_dataset = torchvision.datasets.MNIST(root = './data', train = False, transform = transforms.Compose([   
        transforms.Resize((32,32)),
        transforms.ToTensor(), 
        transforms.Normalize(mean = (0.1325,), std = (0.3105,))
        ]))

# 生成data_loader
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size, 
                                           shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset, 
                                           batch_size = batch_size, 
                                           shuffle = True)

# 构建LeNet5
class ConvNeuralNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeuralNet, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
                nn.BatchNorm2d(6), 
                nn.ReLU(), 
                nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
                nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                nn.BatchNorm2d(16), 
                nn.ReLU(), 
                nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

# 模型调用
model = ConvNeuralNet(num_classes).to(device)

# 定义损失函数和优化器
cost = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

# 训练
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 将数据迁移到GPU上
        images = images.to(device)
        labels = labels.to(device)
        # 调用模型产生结果
        outputs = model(images)
        loss = cost(outputs, labels)
        # 反向传播和优化参数
        # 梯度清零
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 参数优化
        optimizer.step()

        # 打印日志
        if (i + 1) % 400 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

with torch.no_grad():
    correct = 0 
    total = 0 
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))




