from model.config import Config
from model.seq2seq_model import Chatbot
import os
import sys

dataset_save_dir = os.path.join("dataset", "data", "cornell")

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
