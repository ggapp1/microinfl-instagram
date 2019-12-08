import torch
import torch.nn.functional as F
from torch import optim
from torch_geometric.nn import GATConv
from torch.autograd import Variable
from dataset import generate_dataset, split_dataset, Node, ContrastiveLoss
import numpy as np
import random
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from torch_geometric.data import Data
import pickle
import sys

class GAT(torch.nn.Module):
    def __init__(self, num_features, size_emb):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, 32, heads=32, dropout=0.1)
        self.conv2 = GATConv(
            32 * 32, size_emb, heads=1, concat=True, dropout=0.1)

    def forward(self, data):
        x = F.dropout(data.x, p=0.1, training=self.training)
        x = F.elu(self.conv1(x, data.edge_index))
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, data.edge_index)
        return x

def test_model(model, test_dataloader, data):
	ypred = []
	ytest = []
	print('\nTesting...                 ')
	for dataloader in test_dataloader:
		node0, node0_features, node1, node1_features, label = dataloader
		output = model(data)
		output = F.cosine_similarity(output[node0], output[node1])
		output = output.item()
		label = label.item()
		output = 1 if (output>=0) else -1
		ypred.append(output)
		ytest.append(label)

	print('Accuracy: {}'.format(accuracy_score(ytest,ypred)))
	print('F1-sdcore: {}'.format(f1_score(ytest,ypred)))
	print('Avg precision score: {}'.format(average_precision_score(ytest,ypred)))

def train_model(model, train_dataloader, test_dataloader, data):
	criterion = torch.nn.CosineEmbeddingLoss()
	optimizer = optim.Adam(model.parameters(),lr = 0.0001)
	number_epochs = 10
	print('\nTraining...                 ')
	for epoch in range(number_epochs):
		for i, dataloader in enumerate(train_dataloader,0):
			node0, node0_features, node1, node1_features, label = dataloader
			node0_features = node0_features.cuda() 
			node1_features = node1_features.cuda() 
			label = label.cuda() 
			output = model(data)
			optimizer.zero_grad()
			loss = criterion(output[node0], output[node1], label.float())
			loss.backward()
			optimizer.step()
		print("Epoch number {}. Current loss {}\r".format(epoch+1,loss.item()))

	return model


def main():
	if(len(sys.argv) < 4) :
		print('Usage : python gcn.py graphfile no_nodes no_features')
		exit()

	graph, no_nodes, no_features = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])	
	dataset, graph_features, edge_index, gat_features = generate_dataset(graph, no_nodes, no_features)
	num_features = len(graph_features[0][1])
	#these parameters can be changed
	size_emb = 64
	batch_size = 64
	val_split = .2

	train_dataloader, val_dataloader = split_dataset(dataset, batch_size, val_split)
	data = Data(x=torch.tensor(gat_features).float().cuda() , edge_index=edge_index.cuda() , num_nodes=no_nodes)
	model = GAT(num_features, size_emb).cuda() 
	model = train_model(model,train_dataloader, val_dataloader, data)

	test_model(model, val_dataloader, data)
	
if __name__ == '__main__':
	main()