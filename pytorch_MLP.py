import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_

class MLP_torch(nn.Module):

    def __init__(self,layer_sizes):
        super().__init__()
        # 定义一下每层的大小，是否使用偏置项，神经元激活函数，输出层激活函数
        self.layer_sizes = layer_sizes#一个包含每层神经元个数的列表
        self.activation=torch.relu #神经元的激活函数是Relu
        self.out_activation=lambda x: x #输出的激活函数先不加
        self.layers = nn.ModuleList()#用列表的形式存储每个神经元层

        '''
        接下去创建连接层结构        
        '''
        num_in = self.layer_sizes[0]
        for num_out in layer_sizes[1:]:
            #这里每层神经元用torch.Liner创建，然后追加到列表，明确是否使用偏执项
            self.layers.append(nn.Linear(num_in,num_out,bias=True))
            #然后对这层参数进行初始化
            normal_(self.layers[-1].weight,std=1)
            #偏置项之前虽然加了，但是这里全部记作0
            self.layers[-1].bias.data.fill_(0)
            num_in = num_out

    #传入训练数据
    def forward(self,x):
        '''
        forward函数是我们重新定义的，训练时会自动调用
        这里定义了一下前向传播
        '''
        #先经过输入层和隐层，-1是防止经过输出层，输出层激活函数不一样所以单独经过
        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            x = self.activation(x)
        #经过输出层
        x = self.layers[-1](x)
        x = self.out_activation(x)
        return x

''' 
设置超参数
'''
num_epochs = 200 #训练轮数
learning_rate = 0.05 #学习率
batch_size = 128 #批大小
eps = 1e-7 #一个很小的值，防止在计算交叉熵时log出现问题，比如除数为0
#torch.manual_seed(0)

#初始化MLP模型
#这里是3层(包括输入输出层)，输入层神经元个数和特征一致，输出层个数个和类别个数一致
mlp = MLP_torch(layer_sizes=[784,256,128,10])

#SGD优化器
opt = torch.optim.SGD(mlp.parameters(),lr=learning_rate)

#导入数据集
train_data = np.loadtxt('mnist_train.csv',delimiter=',')
test_data = np.loadtxt('mnist_train.csv',delimiter=',')
x_train = train_data[:,:train_data.shape[1]-1]
y_train = train_data[:,-1].reshape(-1,1)
x_test = test_data[:,:test_data.shape[1]-1]
y_test = test_data[:,-1].reshape(-1,1)


#开始训练
losses = [] #记录训练损失
test_losses = [] #记录测试损失
test_accs = [] #记录准确率
for epoch in range(num_epochs):
    print('正在训练第{}轮'.format(epoch))
    st = 0
    loss = []
    while 1:
        ed = min(st +batch_size,len(x_train))
        if st >=ed:
            break
        #取出一批，转为张量
        x = torch.tensor(x_train[st:ed],dtype=torch.float32)
        y = torch.tensor(y_train[st:ed],dtype=torch.long).flatten()#标签需要整数
        #计算MLP的预测,他会自动调用forward函数
        y_pred = mlp(x)
        #y_pred = F.softmax(y_pred,dim=1)
        #计算交叉熵损失函数,神经网络中一般求均值不是和
        loss1 = nn.CrossEntropyLoss()
        train_loss = loss1(y_pred,y)
        #清空梯度
        opt.zero_grad()
        #反向传播
        train_loss.backward()
        #更新参数
        opt.step()

        loss.append(train_loss.detach().numpy())#detach()去掉也不影响，但是保留的化意图更明确(不再使用其中梯度)
        st += batch_size

    losses.append(np.mean(loss))

    #计算测试集上的交叉熵，不需要计算梯度的部分可以用torch.inference_mode()加速
    #一个管理器，优化内存之类的
    with torch.inference_mode():
        x = torch.tensor(x_test,dtype=torch.float32)
        y = torch.tensor(y_test,dtype=torch.long).flatten()#标签需要整数
        y_pred = mlp(x)
        loss2 = nn.CrossEntropyLoss()
        test_loss = loss2(y_pred,y)
        y_pred_softmax = F.softmax(y_pred,dim=1)
        max_index = torch.argmax(y_pred_softmax,dim=1)
        test_acc = torch.sum(max_index == y) / len(x_test)
        test_losses.append(test_loss.detach().numpy())
        test_accs.append(test_acc.detach().item())

print('测试准确率：',test_accs[-1])

plt.figure(figsize=(16,6))
plt.subplot(121)
plt.plot(losses,color='blue',label='train_loss')
plt.plot(test_losses,'r--',label='test_loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Train_Test_Loss')
plt.legend()

plt.subplot(122)
plt.plot(range(num_epochs),test_accs,color='red')
plt.ylim([0.0,1.0])
plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.show()