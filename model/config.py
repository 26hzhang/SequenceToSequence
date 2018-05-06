import os
from model.logger import get_logger
from model.data_utils import load_data


class Config:
    def __init__(self, data_path):
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        self.logger = get_logger(os.path.join(self.ckpt_path, "log.txt"))
        metadata = load_data(data_path)
        self.vocab, self.word_dict = metadata["vocab"], metadata["dict"]
        del metadata
        self.vocab_size = len(self.vocab)
        self.rev_word_dict = dict([(idx, word) for word, idx in self.word_dict.items()])

    # hyperparameters
    grad_clip = 5.0  # gradient clip
    lr = 0.0001  # learning rate
    keep_prob = 0.6  # dropout keep probability
    emb_dim = 300  # word embedding size
    num_units = 256  # number of units for RNN cells
    num_layers = 2  # number of RNN layers
    attention = "Bahdanau"  # attention mechanism: None, "Bahdanau", "Luong"
    use_beam_search = True  # True to use beam search mechanism (only for decode)
    beam_size = 5  # beam size
    batch_size = 32  # batch size
    epochs = 30  # training epochs
    max_to_keep = 5  # maximal number of trained model to be kept

    # paths
    ckpt_path = "ckpt/"
    summary_dir = "summary/"
