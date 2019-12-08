import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from dataset import generate_dataset, split_dataset, Node, ContrastiveLoss
import numpy as np
import random
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import pickle

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
		return F.sigmoid(self.fc5(x))


def test_model(model, test_dataloader, test=1):
	ypred = []
	ytest = []
	ypred_raw = []
	i = 0
	for data in test_dataloader:
		node0, node0_features, node1, node1_features, label = data
		output = model(Variable(node0_features) ,Variable(node1_features) )
		output = output.item()
		ypred_raw.append(output)
		
		label = label.item()
		if(output>=0.5):
			output = 1
		else:
			output = 0
		ypred.append(output)
		ytest.append(label)


		i = i + 1
		if(test == 0 and i >= 100):
			break

	print('accuracy {}'.format(accuracy_score(ytest,ypred)))
	print('f1 {}'.format(f1_score(ytest,ypred)))
	print('avg precision score {}'.format(average_precision_score(ytest,ypred)))
	print('---')
	return ytest, ypred, ypred_raw

def train_model(model, train_dataloader, test_dataloader):
	criterion = nn.BCELoss()
	optimizer = optim.Adam(model.parameters(),lr = 0.0001)
	number_epochs = 20

	for epoch in range(number_epochs):
		for i, data in enumerate(train_dataloader,0):
			node0, node0_features, node1, node1_features, label = data
	#		label[label!=1] = 0
			node0_features = node0_features 
			node1_features = node1_features 
			label = label 
			optimizer.zero_grad()
			output= model.forward(node0_features, node1_features)
			loss = criterion(output.T, label.float())
			loss.backward()
			optimizer.step()

		print("Epoch number {}. Current loss {}\r".format(epoch,loss.item()))
	#	test_model(model, test_dataloader, 0)
	return model




def main():

	#features 1
	print('############')
	print('features 1')
	dataset, graph_features, edge_index, features = generate_dataset('wiki', 10000, 1)
	num_features = len(graph_features[0][1])
	size_emb = 64
	batch_size = 128
	val_split = .2

	train_dataloader, val_dataloader = split_dataset(dataset, batch_size, val_split)
	model = CE(num_features, size_emb) 
	model = train_model(model, train_dataloader, val_dataloader)

	ytest1, ypred1, ypred_raw1 = test_model(model, val_dataloader)

	#features 2
	print('\n############')
	print('features 2')
	dataset, graph_features, edge_index, features = generate_dataset('wiki', 10000, 2)
	num_features = len(graph_features[0][1])

	train_dataloader, val_dataloader = split_dataset(dataset, batch_size, val_split)
	model = CE(num_features, size_emb) 
	model = train_model(model, train_dataloader, val_dataloader)

	ytest2, ypred2, ypred_raw2 = test_model(model, val_dataloader)

	#features 3
	print('\n############')
	print('features 3')
	dataset, graph_features, edge_index, features = generate_dataset('wiki', 10000, 3)
	num_features = len(graph_features[0][1])

	train_dataloader, val_dataloader = split_dataset(dataset, batch_size, val_split)
	model = CE(num_features, size_emb) 
	model = train_model(model, train_dataloader, val_dataloader)

	ytest3, ypred3, ypred_raw3 = test_model(model, val_dataloader)

	predictions = [[ytest1, ypred1, ypred_raw1],[ytest2, ypred2, ypred_raw2], [ytest3, ypred3, ypred_raw3]]

	outfile = open('cross_pred_wiki','wb')
	pickle.dump(predictions, outfile)


if __name__ == '__main__':
	main()