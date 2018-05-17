# Sequence to Sequence Learning

![](https://img.shields.io/badge/Python-3.6.5-brightgreen.svg) ![](https://img.shields.io/badge/Tensorflow-1.8.0-yellowgreen.svg)

This repository builds a sequence to sequence learning algorithm with attention mechanism, which aims to tackle some practical tasks such as simple dialogue, machine translation, pronounce to word and etc. This repo is implemented by tensorflow.

### Usage
Before starting the experiment, you need to pull the data first (_use cornell dataset as example_):
```bash
$ cd dataset
$ bash down_cornell.sh
```
It will create a directory `raw/cornell`, and the downloaded raw data will be stored under this directory.  
**Note**: other `.sh` data pullers will download and unzip data into `raw/` folder as a sub-directory with a specific name.

Then go back tp the repository root, and execute the following commands to start a training or inference task:
```bash
$ cd ..
$ python3 cornell_dialogue.py --mode train  # or decode if you have pretrained checkpoints
```
It will cleanup the dataset, create vocabularies + train/test dataset indices and save the processed data to `dataset/data/cornell` directory (If the processed data already exists, will skip this process).  
Then load the pre-setup configurations (you can change the configurations in the python file), and create the model and start a training session.
```bash
No preprocessed dataset found, create from cornell raw data...
Read cornell movie lines: 304713it [00:02, 128939.96it/s]
Read cornell movie conversations: 83097it [00:01, 46060.47it/s]
Create cornell utterance pairs: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 83097/83097 [01:02<00:00, 1319.20it/s]
Build vocabulary: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 158669/158669 [00:02<00:00, 77018.89it/s]
Build dataset: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 158669/158669 [00:01<00:00, 89873.72it/s]
Load configurations...
Load dataset and create batches...
Prepare train batches: 4711it [00:02, 2225.27it/s]
Prepare test batches: 248it [00:00, 3951.25it/s]
Building model...
source embedding shape: [None, None, 1024]
target input embedding shape: [None, None, 1024]
bi-directional rnn output shape: [None, None, 2048]
encoder input projection shape: [None, None, 1024]
encoder output shape: [None, None, 1024]
decoder rnn output shape: [None, None, 10004] (last dimension is vocab size)
number of trainable parameters: 78197524.
Start training...
Epoch 1 / 60:
   1/4711 [..............................] - ETA: 1468s - Global Step: 1 - Train Loss: 9.2197 - Perplexity: 10094.0631
...
```

### Datasets
List of datasets that the mode of this repository is able to handle.

- [x] [Cornell Movie--Dialogs Corpus](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html).
- [x] Twitter Chat, borrowed from [[marsan-ma/chat_corpus]](https://github.com/Marsan-Ma/chat_corpus/), with 700k lines tweets, where odd lines are tweets and even lines are responded tweets.
- [x] [CMU Pronouncing Dictionary](http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b).
- [ ] [IWSLT 2012 MT Track](http://hltc.cs.ust.hk/iwslt/index.php/evaluation-campaign/ted-task.html#MTtrack) dataset, _English-French translation_.
- [ ] [IWSLT Evaluation 2016 MT Track](https://sites.google.com/site/iwsltevaluation2016/mt-track) dataset, _English-French translation_.
- [x] [Europarl](http://www.statmt.org) dataset, _English-French translation_, reference: [[How to Prepare a French-to-English Dataset for Machine Translation]](https://machinelearningmastery.com/prepare-french-english-dataset-machine-translation/).

### Implementation List
- [x] Build basic model.
- [x] Add Bahdanau and Luong attention.
- [x] Add dropout wrapper
- [x] Add residual wrapper.
- [x] Add learning rate decay.
- [x] Add different training optimizers.
- [x] Add bidirectional rnn for encoder.
- [ ] Add sub-word module, ref: [[BPE]](https://github.com/rsennrich/subword-nmt).
- [ ] Add GNMTAttentionMultiCell wrapper, ref: [[Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation]](https://arxiv.org/pdf/1609.08144.pdf). source: [[tensorflow/nmt/nmt/gnmt_model.py]](https://github.com/tensorflow/nmt/blob/master/nmt/gnmt_model.py).
- [ ] Add BLEU measurement.

### Reference
- [tensorflow/nmt](https://github.com/tensorflow/nmt).
- [suriyadeepan/practical_seq2seq](https://github.com/suriyadeepan/practical_seq2seq).
- [JayParks/tf-seq2seq](https://github.com/JayParks/tf-seq2seq).
- [google/seq2seq](https://github.com/google/seq2seq).
