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
        return x#F.log_softmax(x, dim=1)

def test_model(model, test_dataloader, test, data):
	ypred = []
	ytest = []
	ypred_raw = []
	i = 0
	for dataloader in test_dataloader:
		node0, node0_features, node1, node1_features, label = dataloader
		output = model(data)
		output = F.cosine_similarity(output[node0], output[node1])
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

def train_model(model, train_dataloader, test_dataloader, data):
	criterion = torch.nn.CosineEmbeddingLoss()
	optimizer = optim.Adam(model.parameters(),lr = 0.0001)
	number_epochs = 10
	#data = data 

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
		print("Epoch number {}. Current loss {}\r".format(epoch,loss.item()))
	#	test_model(model, test_dataloader, 0, data)
	return model


def main():
	#features 1
	print('############')
	print('features 1')
	dataset, graph_features, edge_index, gat_features = generate_dataset('wiki_sampled', 3000, 1)
	num_features = len(graph_features[0][1])
	size_emb = 64
	batch_size = 64
	val_split = .2
	train_dataloader, val_dataloader = split_dataset(dataset, batch_size, val_split)
	data = Data(x=torch.tensor(gat_features).float().cuda() , edge_index=edge_index.cuda() , num_nodes=3000)
	model = GAT(num_features, size_emb).cuda() 
	model = train_model(model,train_dataloader, val_dataloader, data)

	ytest1, ypred1, ypred_raw1 = test_model(model, val_dataloader, 1, data)
	
	#features 1
	print('############')
	print('features 2')
	dataset, graph_features, edge_index, gat_features = generate_dataset('wiki_sampled', 3000, 2)
	num_features = len(graph_features[0][1])
	size_emb = 64
	batch_size = 128
	val_split = .2
	train_dataloader, val_dataloader = split_dataset(dataset, batch_size, val_split)
	data = Data(x=torch.tensor(gat_features).float().cuda() , edge_index=edge_index.cuda() , num_nodes=3000)
	model = GAT(num_features, size_emb) .cuda()
	model = train_model(model,train_dataloader, val_dataloader, data)

	ytest2, ypred2, ypred_raw2 = test_model(model, val_dataloader, 1, data)

	#features 1
	print('############')
	print('features 3')
	dataset, graph_features, edge_index, gat_features = generate_dataset('wiki_sampled', 3000, 3)
	num_features = len(graph_features[0][1])
	size_emb = 64
	batch_size = 128
	val_split = .2
	train_dataloader, val_dataloader = split_dataset(dataset, batch_size, val_split)
	data = Data(x=torch.tensor(gat_features).float().cuda() , edge_index=edge_index.cuda() , num_nodes=3000)
	model = GAT(num_features, size_emb).cuda() 
	model = train_model(model,train_dataloader, val_dataloader, data)

	ytest3, ypred3, ypred_raw3 = test_model(model, val_dataloader, 1, data)

	predictions = [[ytest1, ypred1, ypred_raw1],[ytest2, ypred2, ypred_raw2], [ytest3, ypred3, ypred_raw3]]

	outfile = open('gat_pred_wiki','wb')
	pickle.dump(predictions, outfile)

if __name__ == '__main__':
	main()