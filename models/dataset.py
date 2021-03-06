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

class InstagramDataset(Dataset):
	def __init__(self, graph, features):
		self.graph = graph
		self.features = features
	
	def __getitem__(self, index):
		nodes = random.choice(self.features)
		return torch.tensor(nodes[0]), torch.tensor(nodes[1]).float(), torch.tensor(nodes[2]), torch.tensor(nodes[3]).float(), nodes[4]

	def __len__(self):
		return len(self.graph)

def split_dataset(dataset, batch_size, validation_split):
	"""
	Generates training and validation splits

	Arguments:
	dataset -- A InstagramDataset object dataset, based torch's class  Dataset
	batch_size -- size of the batch of the datasets
	validation_split -- percentage of the dataset that will be used in validation.
	Return:
	train_dataloader -- training torch dataloader 
	test_dataloader -- test torch dataloader 
	"""	

	# Creating data indexes for training and validation splits:
	dataset_size = len(dataset)
	indexes = list(range(dataset_size))
	split = int(np.floor(validation_split * dataset_size))

	np.random.shuffle(indexes)
	train_indexes, val_indexes = indexes[split:], indexes[:split]

	# Creating data samplers and loaders:
	train_sampler = SubsetRandomSampler(train_indexes)
	valid_sampler = SubsetRandomSampler(val_indexes)
	train_dataloader = DataLoader(dataset, batch_size=batch_size, 
						sampler=train_sampler, num_workers=8,)
	validation_dataloader = DataLoader(dataset, batch_size=1,
							sampler=valid_sampler, num_workers=8)

	return train_dataloader, validation_dataloader

def get_features(node_features, no_features): 
	"""
	For a given node, returns its features shaped to the convolutional matrix features

	Arguments:
	node_features -- list of lists containing the features of a node
	no_features -- Which set of features will be used
	Return:
	np array containing the features of a node
	"""	
	if(no_features==1):
		return node_features.embedding
	features = np.concatenate((node_features.features))	
	if(no_features==2):
		return np.concatenate((node_features.embedding, features))
	else:
		walk = np.concatenate((node_features.walk[0], node_features.walk[1], node_features.walk[2]))
		return np.concatenate((node_features.embedding, features, walk))

def sample_graph_features(graph, graph_features, no_edges, no_features=1, siamese=0):
	"""
	Generates sampled nodes to train the models.

	Arguments:
	graph -- graph file used.
	graph_features -- A list where the indexes are the node's id and the values are the node'srepresentation
	no_edges -- no_edges of each class to be sampled
	no_features -- Which set of features will be used. 
	siamese -- 1 If the dataset is for a siamese network, else 0
	Return:
	sampled_graph -- a list with 2*no_edges pairs of nodes (no_edges adjacent and no_edges non adjacent nodes)
	"""	
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
	"""
	Generates the matrix features used on convolutional models.

	Arguments:
	graph -- graph file used.
	graph_features -- A list where the indexes are the node's id and the values are the node'srepresentation
	no_features -- Which set of features will be used. 
	size -- size of the feature array
	Return:
	features -- A special matrix, mode of numpy arrays, of features used on convolutional models, similar to the graph_features
	"""	
	nodes = list(graph.nodes)
	features = np.zeros((len(nodes),size))

	for i in nodes:
		features[i] = get_features(graph_features[i], no_features)
	return features   

def generate_dataset(graph_name, no_edges, no_features, siamese=0):
	"""
	Generates all the necessary data to train the models.

	Arguments:
	graph_name -- Name of the graph file used.
	no_edges -- No. of edges that will be sampled to the dataset
	no_features -- Which set of features will be used. 
	siamese -- 1 If the dataset is for a siamese network, else 0
	Return:
	dataset -- A InstagramDataset object dataset, based torch's class  Dataset
	graph_features -- A list where the indexes are the node's id and the values are a list of lists with node's representation
	edge_index -- A COO adjacency matrix of the graph
	features -- A special matrix of features used on convolutional models, similar to the graph_features
	"""	
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