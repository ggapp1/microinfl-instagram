import pickle
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from dataset import generate_dataset, split_dataset, Node, ContrastiveLoss
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

class SiameseNN(torch.nn.Module):
	def __init__(self, num_features, size_emb):
		super(SiameseNN, self).__init__()
		self.fc1 = nn.Linear(num_features, num_features)
		self.drp = nn.Dropout(p=0.1)
		self.fc2 = nn.Linear(num_features,256)
		self.fcc = nn.Linear(256, 128)
		self.fc3 = nn.Linear(128, 128)
		self.fc4 = nn.Linear(128, 64)
		self.fc5 = nn.Linear(64, size_emb)

	def forward_once(self, x):
		x = F.rrelu(self.fc1(x))
		x = F.rrelu(self.drp(x))
		x = F.rrelu(self.fc2(x))
		x = F.rrelu(self.fcc(x))
		x = F.rrelu(self.fc3(x))
		x = F.rrelu(self.fc4(x))
		return self.fc5(x)

	def forward(self, input1, input2):
		output1 = self.forward_once(input1)
		output2 = self.forward_once(input2)
		return output1, output2

def test_model(model, test_dataloader):
	ypred = []
	ytest = []

	for data in test_dataloader:
		node0, node0_features, node1, node1_features, label = data
		output1, output2 = model(Variable(node0_features).cuda(),Variable(node1_features).cuda())
		output = F.cosine_similarity(output1, output2)
		output = output.item()
		label = label.item()
		output = 1 if (a>=0) else -1
		ypred.append(output)
		ytest.append(label)

	print('Accuracy: {}'.format(accuracy_score(ytest,ypred)))
	print('F1-score: {}'.format(f1_score(ytest,ypred)))
	print('Avg precision score: {}'.format(average_precision_score(ytest,ypred)))

def train_model(model, train_dataloader, test_dataloader):
	criterion = nn.CosineEmbeddingLoss()
	optimizer = optim.Adam(model.parameters(),lr = 0.0001)
	number_epochs = 25

	for epoch in range(number_epochs):
		for i, data in enumerate(train_dataloader,0):
			node0, node0_features, node1, node1_features, label = data
			node0_features = node0_features.cuda()
			node1_features = node1_features.cuda()
			label = label.cuda()
			optimizer.zero_grad()
			output1, output2 = model.forward(node0_features, node1_features)
			loss = criterion(output1, output2, label.float())
			loss.backward()
			optimizer.step()

		print("Epoch number {}. Current loss {}\r".format(epoch,loss.item()))

	return model

def main():
	if(len(sys.argv) < 3) :
		print('Usage : python siamese_nn.py graphfile no_nodes no_features')
		exit()

	graph, no_nodes, no_features = sys.argv[1], int(sys.argv[1]), int(sys.argv[2])	
	dataset, graph_features, edge_index, features = generate_dataset(graph, no_nodes, no_features)

	num_features = len(graph_features[0][1])
	size_emb = 64
	batch_size = 64
	val_split = .2

	train_dataloader, val_dataloader = split_dataset(dataset, batch_size, val_split)
	model = SiameseNN(num_features, size_emb).cuda()
	model = train_model(model, train_dataloader, val_dataloader)

	test_model(model, val_dataloader)


if __name__ == '__main__':
	main()