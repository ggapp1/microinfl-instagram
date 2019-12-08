import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import pickle
import sys
from torch import optim
from torch.autograd import Variable
from dataset import generate_dataset, split_dataset, Node, ContrastiveLoss
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt


class CE(torch.nn.Module):
	def __init__(self, num_features, size_emb):
		super(CE, self).__init__()
		self.fc1 = nn.Linear(num_features*2, num_features)
		self.drp = nn.Dropout(p=0.2)
		self.fc2 = nn.Linear(num_features,64)
		self.fcc = nn.Linear(64, 32)
		self.fc3 = nn.Linear(32, 16)
		self.fc4 = nn.Linear(16, 8)
		self.fc5 = nn.Linear(8, 1)

	def forward(self, node0, node1):
		x = torch.cat((node0, node1),1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.drp(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fcc(x))
		x = F.relu(self.fc3(x))
		x = F.relu(self.fc4(x))
		return torch.sigmoid(self.fc5(x))


def test_model(model, test_dataloader, test=1):
	ypred = []
	ytest = []
	print('\nTesting...                 ')
	for data in test_dataloader:
		node0, node0_features, node1, node1_features, label = data
		output = model(Variable(node0_features) ,Variable(node1_features) )
		output = output.item()
		label = label.item()
		output = 1 if (output>=0.5) else 0
		ypred.append(output)
		ytest.append(label)

	print('Accuracy: {}'.format(accuracy_score(ytest,ypred)))
	print('F1-score: {}'.format(f1_score(ytest,ypred)))
	print('Avg precision score: {}'.format(average_precision_score(ytest,ypred)))


def train_model(model, train_dataloader, test_dataloader):
	criterion = nn.BCELoss()
	optimizer = optim.Adam(model.parameters(),lr = 0.0001)
	number_epochs = 10
	print('\nTraining...             ')
	for epoch in range(number_epochs):
		for i, data in enumerate(train_dataloader,0):
			node0, node0_features, node1, node1_features, label = data
			node0_features = node0_features 
			node1_features = node1_features 
			label = label 
			optimizer.zero_grad()
			output= model.forward(node0_features, node1_features)
			loss = criterion(output.T, label.float())
			loss.backward()
			optimizer.step()

		print("Epoch number {}. Current loss {}\r     ".format(epoch,loss.item()))

	return model




def main():
	if(len(sys.argv) < 4) :
		print('Usage : python cross_class.py graphfile no_nodes no_features')
		exit()

	graph, no_nodes, no_features = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])	
	dataset, graph_features, edge_index, features = generate_dataset(graph, no_nodes, no_features)

	num_features = len(graph_features[0][1])
	size_emb = 64
	batch_size = 128
	val_split = .2

	train_dataloader, val_dataloader = split_dataset(dataset, batch_size, val_split)
	model = CE(num_features, size_emb) 
	model = train_model(model, train_dataloader, val_dataloader)

	test_model(model, val_dataloader)

if __name__ == '__main__':
	main()