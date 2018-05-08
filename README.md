# Chatbot via Sequence to Sequence

![](https://img.shields.io/badge/Python-3.6.5-brightgreen.svg) ![](https://img.shields.io/badge/Tensorflow-1.8.0-yellowgreen.svg)

This is a 

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
number of trainable parameters: 62458644.
Start training...
Epoch 1 / 60:
   1/4711 [..............................] - ETA: 14882s - Global Step: 1 - Train Loss: 9.2197 - Perplexity: 10094.0631
...
```

### Dataset
List of datasets that the mode of this repository is able to handle.

- [x] [Cornell Movie--Dialogs Corpus](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html).
- [x] Twitter Chat, borrowed from [[marsan-ma/chat_corpus]](https://github.com/Marsan-Ma/chat_corpus/), with 700k lines tweets, where odd lines are tweets and even lines are responded tweets.
- [x] [CMU Pronouncing Dictionary](http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b).

### Reference
- [tensorflow/nmt](https://github.com/tensorflow/nmt).
- [suriyadeepan/practical_seq2seq](https://github.com/suriyadeepan/practical_seq2seq).
- [JayParks/tf-seq2seq](https://github.com/JayParks/tf-seq2seq).
- [lc222/seq2seq_chatbot_new](https://github.com/lc222/seq2seq_chatbot_new).
- [marsan-ma/chat_corpus](https://github.com/Marsan-Ma/chat_corpus/).
