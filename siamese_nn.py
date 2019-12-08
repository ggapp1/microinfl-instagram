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





def test_model(model, test_dataloader, test=1):
	ypred = []
	ytest = []
	ypred_raw = []
	i = 0
	for data in test_dataloader:
		node0, node0_features, node1, node1_features, label = data
		output1, output2 = model(Variable(node0_features).cuda(),Variable(node1_features).cuda())
		output = F.cosine_similarity(output1, output2)
		output = output.item()
		ypred_raw.append(output)
		if(output>=0):
			output = 1
		else:
			output = -1

		label = label.item()
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
#		test_model(model, test_dataloader, 0)
	return model




def main():

	#features 1
	print('############')
	print('features 1')
	dataset, graph_features, edge_index, features = generate_dataset('wiki', 10000, 1)
	num_features = len(graph_features[0][1])
	size_emb = 64
	batch_size = 64
	val_split = .2

	train_dataloader, val_dataloader = split_dataset(dataset, batch_size, val_split)
	model = SiameseNN(num_features, size_emb).cuda()
	model = train_model(model, train_dataloader, val_dataloader)

	ytest1, ypred1, ypred_raw1 = test_model(model, val_dataloader)

	#features 2
	print('\n############')
	print('features 2')
	dataset, graph_features, edge_index, features = generate_dataset('wiki', 10000, 2)
	num_features = len(graph_features[0][1])

	train_dataloader, val_dataloader = split_dataset(dataset, batch_size, val_split)
	model = SiameseNN(num_features, size_emb).cuda()
	model = train_model(model, train_dataloader, val_dataloader)

	ytest2, ypred2, ypred_raw2 = test_model(model, val_dataloader)

	#features 3
	print('\n############')
	print('features 3')
	dataset, graph_features, edge_index, features = generate_dataset('wiki', 10000, 3)
	num_features = len(graph_features[0][1])

	train_dataloader, val_dataloader = split_dataset(dataset, batch_size, val_split)
	model = SiameseNN(num_features, size_emb).cuda()
	model = train_model(model, train_dataloader, val_dataloader)

	ytest3, ypred3, ypred_raw3 = test_model(model, val_dataloader)

	predictions = [[ytest1, ypred1, ypred_raw1],[ytest2, ypred2, ypred_raw2], [ytest3, ypred3, ypred_raw3]]

	outfile = open('siamese_pred_wiki','wb')
	pickle.dump(predictions, outfile)


if __name__ == '__main__':
	main()