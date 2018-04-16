from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from nets.Nets import ResCNN
from torch.nn.modules.activation import LogSoftmax

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 5)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(  # @UndefinedVariable
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(  # @UndefinedVariable
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

model = ResCNN(28, 28, 1, 3, 32, 128, 3, 10, 0)
if args.cuda:
    model.cuda()

#optimizer = optim.SGD(model.parameters(), lr=0.042, momentum=0.9, nesterov=True)
optimizer = optim.Adam(model.parameters(), weight_decay=0.0001)

def setLr(opt, lr):
    print("Set LR to %f" % lr)
    for param_group in opt.param_groups:
            param_group['lr'] = lr

def train(epoch):
#     if epoch < 10:
#         setLr(optimizer, 0.0001)
#     
#     if epoch < 5:
#         setLr(optimizer, 0.0005)
#     
#     if epoch < 2:
#         setLr(optimizer, 0.001)
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
#         targetOneHot = torch.zeros(args.batch_size, 10)
#         for ti, t in enumerate(target):
#             targetOneHot[ti, t] = 1.0
#         target = targetOneHot.float()
        
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        
        # this would need some more code to handle the last batch, but the loss looks fine now :)
        #loss = -torch.sum(output * target) / args.batch_size
        loss = F.nll_loss(output, target)
        
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

import numpy as np
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

print(params)

for epoch in range(1, args.epochs + 1):
    train(epoch)
    test()

# def logsoftmax(logits):
#     step1 = torch.exp(logits)
#     print("s1", step1)
#     step2 = torch.sum(step1)
#     print("s2", step2)
#     step3 = torch.log(step2)
#     print("s3", step3)
#     step4 = logits - step3 
#     print("s4", step4)
#     
#     return step4
# 
# def dlogsoftmax(logits):
#     step1 = torch.exp(logits)
#     step2 = torch.Tensor([torch.sum(step1)])
#     step3 = step1 / step2
#     step4 = 1 - step3
#     
#     return step4 
# 
# import torch.nn as nn
# d = torch.Tensor([[    0.1588,   -0.1486,   -0.4741,    0.0360,   -0.1964,    0.2260,    0.1296,   -0.0233,   -0.1085,   -0.0262]])
# s = torch.log(nn.Softmax()(d))
# ls = nn.LogSoftmax()(d)
# ls2 = logsoftmax(Variable(d))
# 
# print(s)
# print(ls)
# print(ls2)
# 
# delta = 0.0001
# pre = logsoftmax(Variable(torch.Tensor([[    0.1588  + delta,   -0.1486,   -0.4741,    0.0360,   -0.1964,    0.2260,    0.1296,   -0.0233,   -0.1085,   -0.0262]])))
# post = logsoftmax(Variable(torch.Tensor([[    0.1588  - delta,  -0.1486,   -0.4741,    0.0360,   -0.1964,    0.2260,    0.1296,   -0.0233,   -0.1085,   -0.0262]])))
# 
# 
# 
# vd = Variable(d, requires_grad=True)
# 
# print("v", logsoftmax(vd))
# 
# vd.backward() # foooooooooooo
# 
# print("grads", vd.grad)
# 
# print("pre", pre)
# print("post", post)
# print("wtf", (pre - post) / (2 * delta))
# 
# 
# dls = dlogsoftmax(d)
# print(dls)
# 
# 
# import numpy as np
# print(np.sum([    1.1721,    0.8619,    0.6225,    1.0366,    0.8217,    1.2536,    1.1384,    0.9770,    0.8972,    0.9742]))
