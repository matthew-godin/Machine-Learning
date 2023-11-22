import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import torchvision.models as models
from PIL import Image
import re
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

results_dir = 'results/'
data_dir = '5_shot/'
train_dir = os.path.join(data_dir, 'train/')
unsorted_test_dir = os.path.join(data_dir, 'test/')
losses_and_accuracies_dir = 'losses_and_accuracies/'
model_path = os.path.join(results_dir, 'model.pth')
optimizer_path = os.path.join(results_dir, 'optimizer.pth')
train_losses_path = os.path.join(losses_and_accuracies_dir, 'train_losses.pkl')
train_accuracies_path = os.path.join(losses_and_accuracies_dir, 'train_accuracies.pkl')
test_losses_path = os.path.join(losses_and_accuracies_dir, 'test_losses.pkl')
test_accuracies_path = os.path.join(losses_and_accuracies_dir, 'test_accuracies.pkl')

# rename train and test datasets directories to avoid alphabetical order conflict
train_directories = os.listdir(train_dir)
for train_directory in train_directories:
  if len(train_directory) == 1:
    os.rename(os.path.join(train_dir, train_directory), os.path.join(train_dir, '0' + train_directory))

# set to True if training additional epochs, set to False if simply want to
# graph data that was already saved
training = True
# set to True if some training has already been made to continue from that training
subsequent = False

# number of epochs already done and saved
n_epochs_done = 0
# total number of epochs we intend to have completed after this training
n_epochs = 70
batch_size_train = 4
batch_size_test = 4
learning_rate = 0.01
momentum = 0.9
log_interval = 10
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.ImageFolder(train_dir, transform=torchvision.transforms.Compose([
                               #torchvision.transforms.Grayscale(),
                               torchvision.transforms.Resize(256),
                               torchvision.transforms.CenterCrop(224),
                               torchvision.transforms.ToTensor()
                             ])),
  batch_size=batch_size_train, shuffle=True)
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
    # Conv(1, 64, 3, 1, 1)
    self.conv1 = nn.Conv2d(1, 64, 3, 1, 1)
    # Conv(64, 128, 3, 1, 1)
    self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
    # Conv(128, 256, 3, 1, 1)
    self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
    # Conv(256, 256, 3, 1, 1)
    self.conv4 = nn.Conv2d(256, 256, 3, 1, 1)
    # Conv(256, 512, 3, 1, 1)
    self.conv5 = nn.Conv2d(256, 512, 3, 1, 1)
    # Conv(512, 512, 3, 1, 1)
    self.conv6 = nn.Conv2d(512, 512, 3, 1, 1)
    # BatchNorm(64)
    self.batch1 = nn.BatchNorm2d(64)
    # BatchNorm(128)
    self.batch2 = nn.BatchNorm2d(128)
    # BatchNorm(256)
    self.batch3 = nn.BatchNorm2d(256)
    # BatchNorm(512)
    self.batch4 = nn.BatchNorm2d(512)
    # Linear(512, 4096)
    #self.fc1 = nn.Linear(512, 4096)
    self.fc1 = nn.Linear(8192, 8192)
    # Linear(4096, 4096)
    #self.fc2 = nn.Linear(4096, 4096)
    self.fc2 = nn.Linear(8192, 8192)
    # Linear(4096, 10)
    #self.fc3 = nn.Linear(4096, 10)
    #self.fc3 = nn.Linear(4096, 22)
    self.fc3 = nn.Linear(8192, 22)
    # Dropout(0.5) (nn.Dropout() is nn.Dropout(0.5) by default)
    self.dropout = nn.Dropout()

  def forward(self, x):
    # Conv(1, 64, 3, 1, 1) - BatchNorm(64) - ReLU - MaxPool(2, 2)
    x = F.max_pool2d(F.relu(self.batch1(self.conv1(x))), 2)
    # Conv(64, 128, 3, 1, 1) - BatchNorm(128) - ReLU - MaxPool(2, 2)
    x = F.max_pool2d(F.relu(self.batch2(self.conv2(x))), 2)
    # Conv(128, 256, 3, 1, 1) - BatchNorm(256) - ReLU
    x = F.relu(self.batch3(self.conv3(x)))
    # Conv(256, 256, 3, 1, 1) - BatchNorm(256) - ReLU - MaxPool(2, 2)
    x = F.max_pool2d(F.relu(self.batch3(self.conv4(x))), 2)
    # Conv(256, 512, 3, 1, 1) - BatchNorm(512) - ReLU
    x = F.relu(self.batch4(self.conv5(x)))
    # Conv(512, 512, 3, 1, 1) - BatchNorm(512) - ReLU - MaxPool(2, 2)
    x = F.max_pool2d(F.relu(self.batch4(self.conv6(x))), 2)
    # Conv(512, 512, 3, 1, 1) - BatchNorm(512) - ReLU
    x = F.relu(self.batch4(self.conv6(x)))
    # Conv(512, 512, 3, 1, 1) - BatchNorm(512) - ReLU - MaxPool(2, 2)
    x = F.max_pool2d(F.relu(self.batch4(self.conv6(x))), 2)
    x = torch.flatten(x, 1)
    # Linear(512, 4096) - ReLU - Dropout(0.5)
    x = self.dropout(F.relu(self.fc1(x)))
    # Linear(4096, 10) - ReLU - Dropout(0.5)
    x = self.dropout(F.relu(self.fc2(x)))
    # Linear(4096, 10)
    x = self.fc3(x)
    return x

def train(epoch):
  network.train()
  train_loss = 0
  correct = 0
  for batch_idx, (data, target) in enumerate(train_loader):
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

#network = Net()
network = models.resnext50_32x4d(pretrained=True)
optimizer = optim.Adam(network.parameters())

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

my_transform = torchvision.transforms.Compose([
                               #torchvision.transforms.Grayscale(),
                               torchvision.transforms.Resize(256),
                               torchvision.transforms.CenterCrop(224),
                               torchvision.transforms.ToTensor()
                               ])

# Generate submission.csv
print()
print("submission.csv generation in progress . . . ")
print()

submission_string = "id,category\n"
submission_id = 0
test_dir_actuals = sorted_alphanumeric(os.listdir(unsorted_test_dir))
for test_dir_actual in test_dir_actuals:
  my_actual_path = os.path.join(unsorted_test_dir, test_dir_actual)
  my_img = Image.open(my_actual_path)
  my_input = my_transform(my_img)
  my_input = my_input.unsqueeze(0)
  network.eval()
  my_output = network(my_input)
  my_pred = my_output.data.max(1, keepdim=True)[1].tolist()[0][0]
  submission_string += str(submission_id) + "," + str(my_pred) + "\n"
  submission_id += 1
submission_file = open("submission.csv", "w")
submission_file.write(submission_string)
submission_file.close()

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