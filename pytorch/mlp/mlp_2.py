import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import os
import pickle

results_dir = 'results/'
mnist_dir = 'mnist/'
losses_and_accuracies_dir = 'losses_and_accuracies/'
model_path = os.path.join(results_dir, 'model.pth')
optimizer_path = os.path.join(results_dir, 'optimizer.pth')
train_losses_path = os.path.join(losses_and_accuracies_dir, 'train_losses.pkl')
train_accuracies_path = os.path.join(losses_and_accuracies_dir, 'train_accuracies.pkl')
test_losses_path = os.path.join(losses_and_accuracies_dir, 'test_losses.pkl')
test_accuracies_path = os.path.join(losses_and_accuracies_dir, 'test_accuracies.pkl')

training = False
subsequent = True

n_epochs_done = 0
n_epochs = 20
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.9
log_interval = 10
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST(mnist_dir, train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ])),
  batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST(mnist_dir, train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ])),
  batch_size=batch_size_test, shuffle=True)
criterion = nn.CrossEntropyLoss()

def save_list(list_to_save, file_path):
  f = open(file_path, "wb")
  pickle.dump(list_to_save, f)
  f.close()

def load_list(file_path):
  f = open(file_path, "rb")
  loaded_list = pickle.load(f)
  f.close()
  return loaded_list

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

if subsequent:
  train_losses = load_list(train_losses_path)
  train_accuracies = load_list(train_accuracies_path)
  test_losses = load_list(test_losses_path)
  test_accuracies = load_list(test_accuracies_path)

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    # BatchNorm(512)
    self.batch1 = nn.BatchNorm1d(512)
    # Linear(784, 512)
    self.fc1 = nn.Linear(784, 512)
    # Linear(512, 512)
    self.fc2 = nn.Linear(512, 512)
    # Linear(512, 10)
    self.fc3 = nn.Linear(512, 10)

  def forward(self, x):
    x = torch.flatten(x, 1)
    # Linear(784, 512) - ReLU
    x = F.relu(self.fc1(x))
    # Linear(512, 512) - BatchNorm(512) - ReLU
    x = F.relu(self.batch1(self.fc2(x)))
    # Linear(512, 512) - BatchNorm(512) - ReLU
    x = F.relu(self.batch1(self.fc2(x)))
    # Linear(512, 10)
    x = self.fc3(x)
    return x

def train(epoch):
  network.train()
  train_loss = 0
  correct = 0
  for batch_idx, (data, target) in enumerate(train_loader):
    data = data.view(data.shape[0], -1)
    optimizer.zero_grad()
    output = network(data)
    loss = criterion(output, target)
    loss.backward()
    train_loss += loss.item()
    pred = output.data.max(1, keepdim=True)[1]
    correct += pred.eq(target.data.view_as(pred)).sum()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      torch.save(network.state_dict(), model_path)
      torch.save(optimizer.state_dict(), optimizer_path)
  train_losses.append(train_loss)
  train_accuracies.append(100. * correct / len(train_loader.dataset))
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    train_loss / len(train_loader.dataset), correct, len(train_loader.dataset),
    100. * correct / len(train_loader.dataset)))

def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data = data.view(data.shape[0], -1)
      output = network(data)
      test_loss += criterion(output, target).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_losses.append(test_loss)
  test_accuracies.append(100. * correct / len(test_loader.dataset))
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss / len(test_loader.dataset), correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

if not os.path.isdir(results_dir):
  os.mkdir(results_dir)
open(model_path, 'a').close()
open(optimizer_path, 'a').close()

if subsequent:
  network_state_dict = torch.load(model_path)
  network.load_state_dict(network_state_dict)
  optimizer_state_dict = torch.load(optimizer_path)
  optimizer.load_state_dict(optimizer_state_dict)

if training:
  for epoch in range(n_epochs_done + 1, n_epochs + 1):
    train(epoch)
    test()

if not os.path.isdir(losses_and_accuracies_dir):
  os.mkdir(losses_and_accuracies_dir)
open(train_losses_path, 'a').close()
open(train_accuracies_path, 'a').close()
open(test_losses_path, 'a').close()
open(test_accuracies_path, 'a').close()

if training:
  save_list(train_losses, train_losses_path)
  save_list(train_accuracies, train_accuracies_path)
  save_list(test_losses, test_losses_path)
  save_list(test_accuracies, test_accuracies_path)

x_labels = list(range(1, n_epochs + 1))
#plt.scatter(x_labels, train_losses, color='blue')
plt.plot(x_labels, train_accuracies)
#plt.scatter(x_labels, test_losses, color='red')
#plt.plot(x_labels, test_accuracies)
plt.xlabel('Number of Epochs')
plt.ylabel('Training Accuracy')
#plt.ylabel('Cross-Entropy Loss')
plt.show()