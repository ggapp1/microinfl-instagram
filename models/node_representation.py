import networkx as nx
import pickle
import pandas as pd
import numpy as np
import random
from gensim.models import Word2Vec
from nltk.tokenize import RegexpTokenizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import sys

class Node:
	def __init__(self, node, embedding, features, walk):
		self.node = node
		self.embedding = embedding
		self.features = features
		self.walk = walk

def load_node2vec(node2vec_file):
	"""
	loads a file that contains the node2vec embeddings for the graph

	Arguments:
	node2vec_file -- the node2vec file, where eac line is constituted by nodeid node2vec_embedding
	returns:
	node2vec -- dict containing the node2vec embeddings
	"""
	node2vec = {}
	print('# loading node2vec...', end='\r')
	with open(node2vec_file) as file:
		next(file)
		for line in file:
			l = [float(i) for i in line.split()]
			node2vec[int(l[0])] = l[1:]
	print('# node2vec ok		   \n')

	return node2vec		

def print_top_words(model, feature_names, n_top_words):
	topics=[]
	for topic_idx, topic in enumerate(model.components_):
		topics.append([feature_names[i]
		for i in topic.argsort()[:-n_top_words - 1:-1]])
	topics = [t[0] for t in topics]
	return topics

def lda_text(text):
	number_topics = 3
	no_top_words = 1
	no_features = 10
	# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
	ct_vectorizer = CountVectorizer(max_features=no_features)
	text_ct = ct_vectorizer.fit_transform(text)

	lda = LatentDirichletAllocation(n_components=number_topics, n_jobs=-1)
	lda.fit(text_ct)
	
	ct_features_names = ct_vectorizer.get_feature_names()
	return print_top_words(lda, ct_features_names, no_top_words)

def load_text(text_file, size_w2v):
	file = open(text_file, 'rb')
	users_text = pickle.load(file)
	file.close()
	tokenizer = RegexpTokenizer(r'\w+')

	df_user_text = pd.DataFrame(list(users_text.items()), columns=['username', 'text'])
	df_user_text['text'] = df_user_text['text'].apply(lambda x: tokenizer.tokenize(str(x)))

	text = df_user_text['text'].tolist()
	print('# running w2v...', end='\r')
	w2v_model = Word2Vec(text, size=size_w2v, window=5, min_count=1, workers=6, max_vocab_size=200000)
	print("# w2v ok		        	\n")

	return df_user_text, text, w2v_model

def get_topics_w2v(username, size_w2v, df_user_text, text, w2v_model):
	try:
		text = df_user_text['text'][df_user_text['username'] == username].tolist()
		w2v = []
		if(len(text[0]) > 0):
			topics = lda_text(text[0])
			for topic in topics:
				w2v.append(w2v_model.wv[topic])
		else:
			w2v = np.zeros([3,size_w2v])
	except Exception as e:
		w2v = np.zeros([3,size_w2v])
	return w2v


def random_walk_sampling_simple(complete_graph, origin_index, nodes_to_sample):
	sampled_graph = []
	curr_node = origin_index
	while len(sampled_graph) != nodes_to_sample:
		edges = [n for n in complete_graph.neighbors(curr_node)]
		index_of_edge = random.randint(0, len(edges) - 1)
		chosen_node = edges[index_of_edge]
		sampled_graph.append(chosen_node)
		curr_node = chosen_node

	return sampled_graph

def main():
	"""
	generates a pickle file containig the graph features. The ouputfile is named graphname_features
	"""
	if(len(sys.argv) < 6) :
		print('Usage : python node_representation.py graphfile node2vec_file text_file word2vec_size, size_walk')
		exit()

	graph, node2vec_file, text_file, size_w2v, size_walk  = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])	
	#loads graph, word2vec corpora and node2vec model
	file = open(graph, 'rb')
	G = pickle.load(file) 
	file.close()
	node2vec = load_node2vec(node2vec_file)
	df_user_text, text, w2v_model = load_text(text_file, size_w2v)

	G = G.to_directed()
	G =  nx.convert_node_labels_to_integers(G)

	nodes = list(G.nodes)
	graph_data = {}

	print('# iterating... ')
	for node in nodes:
		username = node
		emb = node2vec[node]
		features = get_topics_w2v(username, size_w2v, df_user_text, text, w2v_model)
		walk_nodes = random_walk_sampling_simple(G, node, size_walk)
		walk = []
		for n in walk_nodes:
			walk.append(node2vec[n])   
		graph_data[node] = Node(username, emb, features, walk)

	print('ok!')	
	outfile = open(graph+'_features','wb')
	pickle.dump(graph_data, outfile)

if __name__ == '__main__':
	main()