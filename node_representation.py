import networkx as nx
import pickle
import pandas as pd
import numpy as np
import random
from gensim.models import Word2Vec
from nltk.tokenize import RegexpTokenizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


class Node:
	def __init__(self, node, embedding, features, walk):
		self.node = node
		self.embedding = embedding
		self.features = features
		self.walk = walk


def load_node2vec():
	file = open('sampled_10k', 'rb')
	G = pickle.load(file)

	node2vec = {}
	print('# loading node2vec...', end='\r')
	with open('wiki.emb') as file:
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

def lda_tags(tags):
	number_topics = 3
	no_top_words = 1
	no_features = 10

	# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
	ct_vectorizer = CountVectorizer(max_features=no_features)
	tags_ct = ct_vectorizer.fit_transform(tags)

	lda = LatentDirichletAllocation(n_components=number_topics, n_jobs=-1)
	lda.fit(tags_ct)
	
	ct_features_names = ct_vectorizer.get_feature_names()
	return print_top_words(lda, ct_features_names, no_top_words)

def load_tags(size_w2v):
	file = open('text_test_etc', 'rb')
	users_tags = pickle.load(file)
	file.close()
	tokenizer = RegexpTokenizer(r'\w+')

	df_user_tags = pd.DataFrame(list(users_tags.items()), columns=['username', 'tags'])
	df_user_tags['tags'] = df_user_tags['tags'].apply(lambda x: tokenizer.tokenize(str(x)))
	tags = df_user_tags['tags'].tolist()
	print('# running w2v...', end='\r')
	w2v_model = Word2Vec(tags, size=size_w2v, window=5, 
						min_count=1, workers=6, max_vocab_size=200000)
	print("# w2v ok			\n")
	return df_user_tags, tags, w2v_model

def get_topics_w2v(username, size_w2v, df_user_tags, tags, w2v_model):
	try:
		tags = df_user_tags['tags'][df_user_tags['username'] == username].tolist()
		w2v = []
		if(len(tags[0]) > 0):
			topics = lda_tags(tags[0])
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
	file = open('wiki_sampled', 'rb')
	G = pickle.load(file) 
	file.close()

	graph_data = {}
	i, j = 0, 0
	size_w2v = 128
	size_walk = 3
	sanity = np.zeros([3,size_w2v])
	G = G.to_directed()
	G =  nx.convert_node_labels_to_integers(G)

	nodes1 = list(G.nodes)

	

	node2vec = load_node2vec()
	#df_user_tags, tags, w2v_model = load_tags(size_w2v)
	print('# iterating... ')
	for node in nodes1:
		username = node#G.nodes[node]['username']
		emb = node2vec[node]
		#features = get_topics_w2v(username, size_w2v, df_user_tags, tags, w2v_model)
		features = np.zeros([3,size_w2v])
		walk_nodes = random_walk_sampling_simple(G, node, size_walk)
		walk = []
		for n in walk_nodes:
			walk.append(node2vec[n])   
			
		graph_data[node] = Node(username, emb, features, walk)
		if np.array_equal(features,sanity):
			j = j + 1
		
		if(i%1000 == 0):
			print('*** No. of nodes :' +str(i) + '. No. of nodes without tags :'+str(j), end='\r')

		i = i + 1	
	outfile = open('wiki_sampled_features','wb')
	pickle.dump(graph_data, outfile)

	print('zerados \n')
	print(j)
	print(i)

if __name__ == '__main__':
	main()