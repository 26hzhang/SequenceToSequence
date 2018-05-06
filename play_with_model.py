from model.config import Config
from model.seq2seq_model import Chatbot
from dataset.data_prepro import process_datasets
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

# Create model and load pretrained parameters
chatbot = Chatbot(config, mode='decode')
chatbot.restore_last_session()

# start to infer
sys.stdout.write("Start to play with chatbot, input utterance and the conversation will end while you input `exit`\n> ")
sys.stdout.flush()
sentence = sys.stdin.readline()
while sentence:
    if sentence == "exit":
        break
    responses = chatbot.inference(sentence)
    for response in responses:
        print(response)
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()

