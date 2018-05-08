import os
from model.logger import get_logger
from model.data_utils import load_data


class Config:
    def __init__(self, tf_config):
        # dataset
        self.task_name = tf_config["dataset_name"]
        self.ckpt_path = tf_config["checkpoint_path"]
        self.summary_dir = tf_config["summary_path"]
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        self.logger = get_logger(os.path.join(self.ckpt_path, "log.txt"))
        dict_data = load_data(tf_config["target_vocabulary"])
        self.source_dict, self.target_dict = dict_data["source_dict"], dict_data["target_dict"]
        del dict_data
        self.source_vocab_size = len(self.source_dict)
        self.vocab_size = len(self.target_dict)
        self.rev_target_dict = dict([(idx, word) for word, idx in self.target_dict.items()])
        self.rev_source_dict = dict([(idx, word) for word, idx in self.source_dict.items()])
        # network parameters
        self.cell_type = tf_config["cell_type"]
        self.attention = tf_config["attention"]
        self.only_top_attention = tf_config["only_top_attention"]
        self.num_units = tf_config["num_units"]
        self.num_layers = tf_config["num_layers"]
        self.emb_dim = tf_config["emb_dim"]
        self.use_beam_search = tf_config["use_beam_search"]
        self.beam_size = tf_config["beam_size"]
        self.maximum_iterations = tf_config["maximum_iterations"]
        # training parameters
        self.lr = tf_config["learning_rate"]
        self.grad_clip = tf_config["grad_clip"]
        self.use_dropout = tf_config["use_dropout"]
        self.keep_prob = tf_config["keep_prob"]
        self.batch_size = tf_config["batch_size"]
        self.epochs = tf_config["epochs"]
        self.max_to_keep = tf_config["max_to_keep"]
