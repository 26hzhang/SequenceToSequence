from dataset.data_prepro import process_datasets
from model.data_utils import load_data, batchnize_dataset
from model.config import Config
from model.seq2seq_model import Chatbot
import os


raw_data_dir = os.path.join("dataset", "raw")
dataset_save_dir = os.path.join("dataset", "data")

# process dataset
if not os.listdir(dataset_save_dir):  # if dataset directory is empty
    print("Process raw data...")
    process_datasets(raw_data_dir, dataset_save_dir)

# load dataset
print("Load dataset...")
dataset = load_data(os.path.join(dataset_save_dir, "dataset.json"))
train_set = dataset["train_set"]
test_set = dataset["test_set"]
del dataset

# create configurations
print("Load configurations...")
config = Config(os.path.join(dataset_save_dir, "metadata.json"))

# prepare training
print("Prepare train batches...")
batch_size = 128
train_batches = batchnize_dataset(train_set, batch_size, config.word_dict)

# build model
print("Build model")
chatbot = Chatbot(config)

chatbot.train(train_batches, epochs=30)
