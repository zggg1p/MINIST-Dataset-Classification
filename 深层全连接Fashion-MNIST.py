#引入相关库
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class Batch_Net(nn.Module):#定义网络
    def __init__(self, in_dim, n_hidden_1, n_hidden_2,n_hidden_3,out_dim):
        super(Batch_Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(True))
        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2, n_hidden_3),
            nn.ReLU(True))
        self.layer4 = nn.Sequential(
            nn.Linear(n_hidden_3,out_dim))
    def forward(self, x):
        x = x.view(x.size()[0], -1)#展平
        hidden_1_out = self.layer1(x)
        hidden_2_out = self.layer2(hidden_1_out)
        hidden_3_out = self.layer3(hidden_2_out)
        out=self.layer4(hidden_3_out)
        return out

# 定义超参数
EPOCH = 20   #遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 64      #批处理尺寸(batch_size)
LR = 0.001        #学习率

# 启用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#加载数据集
train_loader = torch.utils.data.DataLoader(  # 加载训练数据
        datasets.FashionMNIST('./data2', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))  # 数据集给出的均值和标准差系数，每个数据集都不同的，都数据集提供方给出的
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(  # 加载训练数据
        datasets.FashionMNIST('./data2', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # 数据集给出的均值和标准差系数，每个数据集都不同的，都数据集提供方给出的
        ])),
        batch_size=BATCH_SIZE, shuffle=True)

model = Batch_Net(28*28,400,300,100,10)  # 实例化一个网络对象
model = model.to(device)

criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
optimizer = optim.Adam(model.parameters(), lr=LR) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰

# 训练
if __name__ == "__main__":
    best_acc = 85  #2 初始化best test accuracy
    print("Start Training, 深层全连接Fashion-Minist!")  # 定义遍历数据集的次数
    with open("MINIST-FASHION(深层全连接)acc.txt", "w") as f:
        with open("MINIST-FASHION(深层全连接)log.txt", "w")as f2:
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                model.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(train_loader, 0):
                    # 准备数据
                    length = len(train_loader)
                    inputs, labels = data
                    inputs, labels = Variable(inputs), Variable(labels)
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    # forward + backward
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()

                # 每训练完一个epoch测试一下准确率
                print("Waiting Test!")
                model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in test_loader:
                        model.eval()
                        images, labels = data
                        images, labels = Variable(images), Variable(labels)
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    print('测试分类准确率为：%.3f%%' % (100 * correct / total))
                    acc = 100. * correct / total
                    # 将每次测试结果实时写入acc.txt文件中
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if acc > best_acc:
                        f3 = open("MINIST-FASHION(深层全连接).txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
            print('Saving model......')
            torch.save(model, 'MINIST-FASHION(深层全连接)_%03d.pth' % (epoch + 1))
            print("Training Finished, TotalEPOCH=%d" % EPOCH)