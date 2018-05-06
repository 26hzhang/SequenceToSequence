from dataset.data_prepro import process_datasets
from model.data_utils import batchnize_dataset
from model.config import Config
from model.seq2seq_model import Chatbot
import os
import sys


raw_data_dir = os.path.join("dataset", "raw")
dataset_save_dir = os.path.join("dataset", "data")

# process dataset
if not os.path.exists(dataset_save_dir) or not os.listdir(dataset_save_dir):  # if dataset directory is empty
    sys.stdout.write("No preprocessed dataset found, create them from raw data...\n")
    sys.stdout.flush()
    process_datasets(raw_data_dir, dataset_save_dir)

# create configurations
sys.stdout.write("Load configurations...\n")
sys.stdout.flush()
config = Config(os.path.join(dataset_save_dir, "metadata.json"))

# prepare training
sys.stdout.write("Load dataset and create batches...\n")
sys.stdout.flush()
dataset_path = os.path.join(dataset_save_dir, "dataset.json")
train_batches, test_batches = batchnize_dataset(dataset_path, config.batch_size, config.word_dict)

# build model and start training
sys.stdout.write("Building model...\n")
sys.stdout.flush()
chatbot = Chatbot(config, mode="train")
chatbot.train(train_batches, test_batches, epochs=config.epochs)
