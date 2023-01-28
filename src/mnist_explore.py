# %%

from __future__ import print_function
import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import ipdb
from utils import weight_init


# https://github.com/pytorch/examples/blob/main/mnist/main.py
class DefaultModel(nn.Module):
    def __init__(self):
        super(DefaultModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.apply(weight_init)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output, x


# deeper model adapted from https://www.kaggle.com/gustafsilva/cnn-digit-recognizer-pytorch
class DeeperModel(nn.Module):
    def __init__(self):
        super(DeeperModel, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 10)
        )
        self.apply(weight_init)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # import ipdb; ipdb.set_trace()

        conv_features = x.view(x.size(0), -1)
        fc_features = self.fc(conv_features)
        x = F.log_softmax(fc_features, dim=1)
        return x, fc_features


# https://nextjournal.com/gkoehler/pytorch-mnist
class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.apply(weight_init)

    def forward(self, x, return_conv_features=True):
        with torch.no_grad():
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            conv_features = x.view(-1, 320)

        x = F.relu(self.fc1(conv_features))
        fc_features = F.dropout(x, training=self.training)
        x = self.fc2(fc_features)
        x = F.log_softmax(x, dim=1)
        if return_conv_features:
            return x, conv_features
        return x, fc_features


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, features = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader, plot=True):
    model.eval()
    test_loss = 0
    correct = 0
    dataset_features = []
    dataset_labels = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, features = model(data)
            dataset_features.append(features)
            dataset_labels.append(target)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    dataset_features = torch.cat(dataset_features, dim=0).cpu().numpy()
    dataset_labels = torch.cat(dataset_labels, dim=0).cpu().numpy()
    if plot:
        plot_features(dataset_features, dataset_labels, n_comp=2)


def plot_features(data, label, n_comp=3):
    X_embedded = TSNE(n_components=n_comp, verbose=1).fit_transform(data)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_aspect('equal', adjustable='box')
    if n_comp == 3:
        ax = fig.add_subplot(projection='3d')

    # cdict = {0: 'red', 1: 'blue', 2: 'green'}

    markers = ['v', 'x', 'o', '.', '>', '<', '1', '2', '3', '4']

    for i, g in enumerate(np.unique(label)):
        ix = np.where(label == g)
        if n_comp == 3:
            ax.scatter(X_embedded[ix, 0], X_embedded[ix, 1], X_embedded[ix, 2], marker=markers[i], label=g, alpha=0.8)
        else:
            ax.scatter(X_embedded[ix, 0], X_embedded[ix, 1], marker=markers[i], label=g, alpha=0.8)

    ax.set_xlabel('Embedding dim 1')
    ax.set_ylabel('Embedding dim 2')
    if n_comp == 3: ax.set_zlabel('Z Label')
    if n_comp == 3: ax.set_zlabel('Z Label')
    ax.legend(fontsize='small', markerscale=2, loc="upper left", ncol=1)
    # fig.savefig("camera_info_sf_adv.svg", dpi=500, format='svg', metadata=None)
    fig.savefig("tsne.png", dpi=500, format='png', metadata=None)
    # plt.show()


# dataclass to hold arguments
@dataclass
class Args:
    batch_size: int = 64
    test_batch_size: int = 1000
    epochs: int = 10
    lr: float = 1.0
    gamma: float = 0.7
    no_cuda: bool = False
    no_mps: bool = False
    dry_run: bool = False
    seed: int = 1
    log_interval: int = 10
    save_model: bool = False


args = Args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
use_mps = not args.no_mps and torch.backends.mps.is_available()

torch.manual_seed(args.seed)

if use_cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

train_kwargs = {'batch_size': args.batch_size}
test_kwargs = {'batch_size': args.test_batch_size}
if use_cuda:
    cuda_kwargs = {'num_workers': 1,
                   'pin_memory': True,
                   'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset1 = datasets.MNIST('../data', train=True, download=True,
                          transform=transform)
dataset2 = datasets.MNIST('../data', train=False,
                          transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

model = SmallNet().to(device)

total_params = sum(param.numel() for param in model.parameters())
print(f'{total_params:,} total parameters.')

optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader, plot=False)
    scheduler.step()
    ...

if args.save_model:
    torch.save(model.state_dict(), "mnist_cnn.pt")
