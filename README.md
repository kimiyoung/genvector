# GenVector

## Introduction

This repo contains the code and the datasets used in the following paper:

[Multi-Modal Bayesian Embeddings for Learning Social Knowledge Graphs](https://arxiv.org/abs/1508.00715)
Zhilin Yang, Jie Tang, William W. Cohen (IJCAI 2016)

We provide an efficient implementation of GenVector, a multi-modal Bayesian embedding model, which learns a shared latent topic space to generate embeddings of
two modalities---social network users and knowledge concepts. GenVector combines the advantages of topic modeling and embeddings, and
outperforms state-of-the-art methods in the task of learning social knowledge graphs. Our algorithm is deployed on [AMiner](http://aminer.org/).

## Datasets

The dataset can be downloaded [here](https://static.aminer.org/lab-datasets/genvector/data.tar.gz) (8.3G). Please extract the compressed file and
put the directory `data` as an immediate sub-directory of `genvector` (the current directory).
```
wget https://static.aminer.org/lab-datasets/genvector/data.tar.gz
tar zxvf data.tar.gz
```

`data` contains the following files:

1. `gen_pair.*.out`: original data extracted from AMiner. We split the file into 8 separate files, indexed from 0 to 7. Each line is formatted 
as follows:

  ```
  <author_id>;<keyword_1>,<word_cnt_1>;<keyword_2>,<word_cnt_2>;...
  ```
  
  which means that the author with `author_id` has publications where `<keyword_i>` occurs `<word_cnt_i>` times. This file is a (small) subset
  of AMiner.

1. `homepage_test.txt`: the ground truth file for the homepage matching experiment. Each line is formated as follows:

  ```
  <author_id>,<keyword_1>,<keyword_2>,...
  ```

  The keywords following an `author_id` are the research interests of the author.
  
1. `lk_test.txt`: the ground truth file for the LinkedIn matching experiment. Each line is formated as follows:

  ```
  <author_id>,<keyword_1>,<keyword_2>,...
  ```
  
  The keywords following an `author_id` are the research interests of the author.
  
1. `sample_id.txt`: the file that contains the samples that we use to train our model. Each line is an `author_id`.

1. `keyword.model*`: keyword embeddings trained on AMiner. The embeddings are stored in the [gensim](http://radimrehurek.com/gensim/install.html) format. The following
script gives an example of loading and using the embeddings:
  ```python
  import gensim
  model = gensim.models.Word2Vec.load('data/keyword.model') # load the model
  keyword = 'machine_learning'
  embedding = model[keyword] # access the embedding of machine_learning
  sim = model.similarity('machine_learning', 'data_mining') # compute the similarity between data_mining and machine_learning
  ```

1. `online.author_word.model*`: author embeddings trained on AMiner, in the same format as the keyword embeddings. Author embeddings can be
accessed using `author_id`.
  ```python
  import gensim
  author_id = '53f42f36dabfaedce54dcd0c' # Jiawei Han's author_id on AMiner
  model = gensim.models.Word2Vec.load('data/online.author_word.model')
  embedding = model[author_id]
  ```

## Preprocessing

You can directly download the preprocessed data files that are prepared for our model [here](https://static.aminer.org/lab-datasets/genvector/my_data.tar.gz) (110M).
Please extract the compressed file and put the directory `my_data` as an immediate sub-directory of `genvector` (the current directory).
```
wget https://static.aminer.org/lab-datasets/genvector/my_data.tar.gz
tar zxvf my_data.tar.gz
```

Otherwise, you can run the script `prep_data.py` to do the preprocessing. Note that `prep_data.py` has dependencies on [gensim](http://radimrehurek.com/gensim/install.html).
Running `prep_data.py` can take up to 25GB memory and 30 minutes. (Therefore directly downloading the above files is suggested.)
```
python prep_data.py
```

## Training

Once you follow the above instructions and have the directories `data` and `my_data` in place, you are ready to train the model. Compile
and run
```
make
./main
```

This will produce a file `model.save.txt` in the directory `my_data`, which contains the saved model parameters.

Please refer to the documentation inside `model.hpp` if you would like to tune the hyper-parameters.

## Inference

After the model training is done, you can use the saved model for inference. Make sure you have `my_data/model.save.txt` in place before you do inference. Compile and run
```
make
./predict
```

This will produce a file `model.predict.txt` in the directory `my_data`, which contains the prediction results given by the model.
The format of each line is as follows
```
<author_id>,<keyword_1>,<keyword_2>,...
```

The keywords are sorted from the highest probabilities to the lowest; i.e., our model "thinks" `<keyword_1>` is more likely to be a research interest than `<keyword_2>`
for the given `author_id`.

## Evaluation

Two evaluation scripts are in the directory `eval`. Assume that `my_data/model.predict.txt` and the directory `data` are ready. Run and compute
the scores
```
cd eval
python eval_homepage.py
python eval_lk.py
```

## Misc

1. The `author_id`'s we use in the data are consistent with AMiner. You can access the author's profile on AMiner with
`https://aminer.org/profile/<author_id>`

1. The input/output file configuration is done in `config.hpp`.

1. Our implementation leverages [fastapprox](https://github.com/Nigh/fastapprox) for efficient computation of `log`, `pow`, and `exp`.

1. We code needs further documentation and command line arguments support. We will update the code repo in the near future. You might also
contribute to the code base if interested :)

1. Star our repo and/or cite our paper if you find it useful :)

