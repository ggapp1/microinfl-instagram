from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
import networkx as nx
import random
import pickle as pk
import torch
import torch.nn.functional as F


class Node:
	def __init__(self, node, embedding, features, walk):
		self.node = node
		self.embedding = embedding
		self.features = features
		self.walk = walk

class ContrastiveLoss(torch.nn.Module):
	"""
	Contrastive loss function.
	Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
	"""
	def __init__(self, margin=2):
		super(ContrastiveLoss, self).__init__()
		self.margin = margin

	def forward(self, output1, output2, label):
		euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
		label = label.float()
		loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
									  (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

		return loss_contrastive

def split_dataset(dataset, batch_size, validation_split):
	# Creating data indices for training and validation splits:
	dataset_size = len(dataset)
	indices = list(range(dataset_size))
	split = int(np.floor(validation_split * dataset_size))

	np.random.shuffle(indices)
	train_indices, val_indices = indices[split:], indices[:split]

	# Creating PT data samplers and loaders:
	train_sampler = SubsetRandomSampler(train_indices)
	valid_sampler = SubsetRandomSampler(val_indices)
	train_dataloader = DataLoader(dataset, batch_size=batch_size, 
						sampler=train_sampler, num_workers=8,)
	validation_dataloader = DataLoader(dataset, batch_size=1,
							sampler=valid_sampler, num_workers=8)

	return train_dataloader, validation_dataloader


def get_features(node_features, no_features): 
	if(no_features==1):
		return node_features.embedding

	features = np.concatenate((node_features.features))	
	if(no_features==2):
		return np.concatenate((node_features.embedding, features))
	else:
		walk = np.concatenate((node_features.walk[0], node_features.walk[1], node_features.walk[2]))
		return np.concatenate((node_features.embedding, features, walk))

def sample_graph_features(graph, graph_features, no_edges, no_features=1, siamese=0):
	sampled_graph = []

	edges = list(graph.edges)
	nodes = list(graph.nodes)
	
	for i in range(no_edges):
		r = np.random.randint(0,len(edges) - 1)
		node1_pos = edges[r][0]
		node2_pos = edges[r][1]

		node1_pos_features = get_features(graph_features[node1_pos], no_features)
		node2_pos_features = get_features(graph_features[node2_pos], no_features)
											 
		sampled_graph.append([node1_pos, node1_pos_features, node2_pos, node2_pos_features, 1]) 
											   
		node1_neg = nodes[np.random.randint(0,len(nodes) - 1)]
		node2_neg = nodes[np.random.randint(0,len(nodes) - 1)]
											   
		while(graph.has_edge(node1_neg, node2_neg)):
			node1_neg = nodes[np.random.randint(0,len(nodes) - 1)]
			node2_neg = nodes[np.random.randint(0,len(nodes) - 1)]

		node1_neg_features = get_features(graph_features[node1_neg], no_features)
		node2_neg_features = get_features(graph_features[node2_neg], no_features)

		neg_edge = -1 if (siamese == 1) else 0
		sampled_graph.append([node1_neg, node1_neg_features, node2_neg, node2_neg_features, neg_edge])
																																			  
	return sampled_graph
	
def gcn_features(graph, graph_features, no_features, size):
	nodes = list(graph.nodes)
	features = np.zeros((len(nodes),size))

	for i in nodes:
		features[i] = get_features(graph_features[i], no_features)
	return features   

class InstagramDataset(Dataset):
	def __init__(self, graph, features):
		self.graph = graph
		self.features = features
	
	def __getitem__(self, index):
		nodes = random.choice(self.features)
		return torch.tensor(nodes[0]), torch.tensor(nodes[1]).float(), torch.tensor(nodes[2]), torch.tensor(nodes[3]).float(), nodes[4]

	def __len__(self):
		return len(self.graph)



def generate_dataset(graph_name, no_edges, no_features, siamese=0):

	print('Generating dataset... ', end='\r')
	file = open(graph_name, 'rb')
	graph = pk.load(file)
	file.close()

	file = open(graph_name+'_features', 'rb')
	full_graph_features = pk.load(file)
	file.close()

	graph = graph.to_directed()
	graph =  nx.convert_node_labels_to_integers(graph)

	graph_features = sample_graph_features(graph, full_graph_features, no_edges, no_features, siamese)
	dataset = InstagramDataset(graph, graph_features)

	edge_index = torch.tensor(list(graph.edges)).t().contiguous()
	features = gcn_features(graph, full_graph_features, no_features, len(graph_features[0][1]))
	print('Dataset ok!             ')
	return dataset, graph_features, edge_index, features							 