# Predicting relationships between users and microinfluencers on instagram


## Usage

```
$ python modelfile.py graphfile no_nodes no_features
```


## Requirements

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements.

```bash
$ pip install -r requirements.txt
```

## Execution example
```
$ python models/cross_class.py dataset/sampled_10k 10000 100
Dataset ok!             

Training...             
      number 1. Current loss 0.6304970383644104
      number 2. Current loss 0.4780827462673187
      number 3. Current loss 0.4162384867668152
      number 4. Current loss 0.3480367660522461
      number 5. Current loss 0.3127545416355133
      number 6. Current loss 0.2813030779361725
      number 7. Current loss 0.2694847285747528
      number 8. Current loss 0.3530244827270508
      number 9. Current loss 0.3521847426891327
      number 10. Current loss 0.24455657601356506

Testing...                 
Accuracy: 0.868
F1-score: 0.8787878787878788
Avg precision score: 0.7881011811292415

```



### Motivation

Using microinfluencers to promote a brand or a product has become a common and effective practice on Instagram. The reach of a post can be estimated reconstructing the network, that can only be partially observed. A model that perform this task with considerable accuracy will higher accuracy will result in better and more useful ads.

### Dataset
The dataset used was created from a crawl composed of european microinfluencers. It consists of, in most part, by information about posts made by these influencers and its comments. With this dataset, we created a nearly bipartite graph user-microinfluencer.

#### Creating the relationships

Using the Instagram's API, one cannot obtain users’ followers and followees, making the reconstruction of the network a non trivial task. The edges were created based on comments and mentions in each post. 


### Node representation

We Proposed three representation strategies:
- One set of features: Each node is represented by its Node2Vec embedding.
- Two set of features: Each node is represented by its Node2Vec embedding, combined with representations of the latent topics of users’ comments, posts and hashtags.
- Three set of features: Each node is represented by its 2-Feature representation, combined with  Node2Vec embeddings of neighbors obtained by predefined  size random walks.

### Algorithms 

- Baselines:
We compared our models with four baselines: Similarity using each of the three node representation proposed and BiNE, a state-of-art embedding algorithm for bipartite graphs.
- Crossentropy Classifier (CE): A Multilayer Perceptron (MLP) classifier trained with pairs of adjacent and non adjacent nodes.

- Graph Convolutional Network (GCN): Based on the archit
ecture proposed by Kipf et al.
-Graph Attention Network (GAT): Based on the architecture proposed by Veličković et al.

- Siamese Neural Network (SNN): A siamese neural network trained with pairs of adjacent and non adjacent nodes.

### Experiments
First, we compared the performance of each model for each one of the representation strategies. Then, we compared the best ones with the baselines, using  two datasets.

![Image a](https://github.com/ggapp1/microinfl-instagram/blob/master/results/CE.jpg)
![Image description](https://github.com/ggapp1/microinfl-instagram/blob/master/results/GAT.jpg)
![Image description](https://github.com/ggapp1/microinfl-instagram/blob/master/results/GCN.jpg)
![Image description](https://github.com/ggapp1/microinfl-instagram/blob/master/results/SNN.jpg)


## License
[MIT](https://choosealicense.com/licenses/mit/)