import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from classifier.utils import data as data_utils
from torch.autograd import Variable

def make_layers():
	layers = []
	in_channels = 1
	conv2d = nn.Conv2d(in_channels, 6, kernel_size=5)
	layers += [conv2d, nn.ReLU(inplace=True)]
	conv2d = nn.Conv2d(6, 16, kernel_size=5)
	layers += [conv2d, nn.ReLU(inplace=True)]
	layers += [nn.MaxPool2d(kernel_size=2, stride=2)]    
	layers += [nn.Linear(16*5*5, 120)]    
	layers += [nn.Linear(120, 84)]    
	layers += [nn.Linear(84, 10)]    
	return nn.Sequential(*layers)
    
class simpleClassifier(nn.Module):
	def __init__(self):
		super(simpleClassifier, self).__init__()
		self.features = make_layers()        
		self.conv1 = nn.Conv2d(1, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16*5*5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)
        
	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16*5*5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x
    
