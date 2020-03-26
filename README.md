# OpenKE-PyTorch

An Open-source Framework for Knowledge Embedding implemented with PyTorch.

More information is available on our website 
[http://openke.thunlp.org/](http://openke.thunlp.org/)

If you use the code, please cite the following [paper](http://aclweb.org/anthology/D18-2024):

```
 @inproceedings{han2018openke,
   title={OpenKE: An Open Toolkit for Knowledge Embedding},
   author={Han, Xu and Cao, Shulin and Lv Xin and Lin, Yankai and Liu, Zhiyuan and Sun, Maosong and Li, Juanzi},
   booktitle={Proceedings of EMNLP},
   year={2018}
 }
```

This package is mainly contributed (in chronological order) by [Xu Han](https://github.com/THUCSTHanxu13), [Yankai Lin](https://github.com/Mrlyk423), [Ruobing Xie](http://nlp.csai.tsinghua.edu.cn/~xrb/), [Zhiyuan Liu](http://nlp.csai.tsinghua.edu.cn/~lzy/), [Xin Lv](https://github.com/davidlvxin), [Shulin Cao](https://github.com/ShulinCao), [Weize Chen](https://github.com/chenweize1998), [Jingqin Yang](https://github.com/yjqqqaq).

## Installation

1. Install PyTorch

2. Clone the OpenKE-PyTorch branch:

	$ git clone -b OpenKE-PyTorch https://github.com/MikeyQiu/OpenKE
	
	$ cd OpenKE
	
	$ cd openke

3. Compile C++ files
	
	$ bash make.sh
	
4. Quick Start

	$ cd ../
	
	$ cp yago2test/train_analogy_YAGO3-10.py ./
	
	$ python train_analogy_YAGO3-10.py

5. If the test works please change the epoch from 1 to 100 in train_analogy_YAGO3-10.py


## Experimental Settings

For each test triplet, the head is removed and replaced by each of the entities from the entity set in turn. The scores of those corrupted triplets are first computed by the models and then sorted by the order. Then, we get the rank of the correct entity. This whole procedure is also repeated by removing those tail entities. We report the proportion of those correct entities ranked in the top 10/3/1 (Hits@10, Hits@3, Hits@1). The mean rank (MRR) and mean reciprocal rank (MRR) of the test triplets under this setting are also reported.

Because some corrupted triplets may be in the training set and validation set. In this case, those corrupted triplets may be ranked above the test triplet, but this should not be counted as an error because both triplets are true. Hence, we remove those corrupted triplets appearing in the training, validation or test set, which ensures the corrupted triplets are not in the dataset. We report the proportion of those correct entities ranked in the top 10/3/1 (Hits@10 (filter), Hits@3(filter), Hits@1(filter)) under this setting. The mean rank (MRR (filter)) and mean reciprocal rank (MRR (filter)) of the test triplets under this setting are also reported.

More details of the above-mentioned settings can be found from the papers [TransE](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf), [ComplEx](http://proceedings.mlr.press/v48/trouillon16.pdf).

For those large-scale entity sets, to corrupt all entities with the whole entity set is time-costing. Hence, we also provide the experimental setting named "[type constraint](https://www.dbs.ifi.lmu.de/~krompass/papers/TypeConstrainedRepresentationLearningInKnowledgeGraphs.pdf)" to corrupt entities with some limited entity sets determining by their relations.

## Experiments

We have provided the hyper-parameters of some models to achieve the state-of-the-art performace (Hits@10 (filter)) on FB15K237 and WN18RR. These scripts can be founded in the folder "./examples/". Up to now, these models include TransE, TransH, TransR, TransD, DistMult, ComplEx. The results of these models are as follows,

|Model			|	WN18RR	|	FB15K237	| WN18RR (Paper\*)| FB15K237  (Paper\*)|
|:-:		|:-:	|:-:  |:-:  |:-:  |
|TransE	|0.512	|0.476|0.501|0.486|
|TransH	|0.507	|0.490|-|-|
|TransR	|0.519	|0.511|-|-|
|TransD	|0.508	|0.487|-|-|
|DistMult	|0.479	|0.419|0.49|0.419|
|ComplEx	|0.485	|0.426|0.51|0.428|
|ConvE		|0.506	|0.485|0.52|0.501|
|RotatE	|0.549	|0.479|-|0.480|
|RotatE (+adv)	|0.565	|0.522|0.571|0.533|


<strong> We are still trying more hyper-parameters and more training strategies (e.g., adversarial training and label smoothing regularization) for these models. </strong> Hence, this table is still in change. We welcome everyone to help us update this table and hyper-parameters.

