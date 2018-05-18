from dataset.data_europarl import cleanup_sentence, process_europarl
from utils import batchnize_dataset, process_batch_data, UNK, PAD, GO, EOS
from model import Config, SequenceToSequence
import os
import sys
import tensorflow as tf
import argparse


def create_configurations():
    # dataset parameters
    tf.flags.DEFINE_string("dataset_name", "europarl", "dataset name")
    tf.flags.DEFINE_string("raw_data_dir", "dataset/raw/europarl", "path to the raw data directory")
    tf.flags.DEFINE_string("save_dir", "dataset/data/europarl", "path to the processed dataset directory")
    tf.flags.DEFINE_string("vocabulary", "dataset/data/europarl/vocabulary.json", "path to the vocabulary")
    tf.flags.DEFINE_string("dataset", "dataset/data/europarl/dataset.json", "path to the train and test datasets")
    tf.flags.DEFINE_integer("max_sent_len", 100, "maximal number of words for each sentence")
    tf.flags.DEFINE_integer("min_sent_len", 4, "minimal number of words for each sentence")
    tf.flags.DEFINE_integer("en_vocab_size", 20000, "english vocabulary size")
    tf.flags.DEFINE_integer("fr_vocab_size", 20000, "french vocabulary size")
    tf.flags.DEFINE_boolean("lower", True, "convert words into lowercase or not")
    tf.flags.DEFINE_boolean("keep_number", False, "keep number characters in dataset or not")
    tf.flags.DEFINE_boolean("fr_to_en", True, "build data for french to english translation [True] or reverse [False]")
    tf.flags.DEFINE_float("train_ratio", 0.9, "split dataset into train and test dataset according to this ratio")
    # network parameters
    tf.flags.DEFINE_string("cell_type", "lstm", "RNN cell for encoder and decoder: [lstm | gru], default: lstm")
    tf.flags.DEFINE_string("attention", "bahdanau", "attention mechanism: [bahdanau | luong], default: bahdanau")
    tf.flags.DEFINE_boolean("top_attention", True, "apply attention mechanism only on the top decoder layer")
    tf.flags.DEFINE_boolean("use_bi_rnn", False, "apply bidirectional RNN before encoder to process input embeddings")
    tf.flags.DEFINE_integer("num_units", 1024, "number of hidden units in each layer")
    tf.flags.DEFINE_integer("num_layers", 2, "number of layers for encoder and decoder")
    tf.flags.DEFINE_integer("emb_dim", 1024, "embedding dimension for encoder and decoder input words")
    tf.flags.DEFINE_boolean("use_beam_search", True, "use beam search strategy for decoder")
    tf.flags.DEFINE_integer("beam_size", 5, "beam size")
    tf.flags.DEFINE_boolean("use_dropout", True, "use dropout for rnn cells")
    tf.flags.DEFINE_boolean("use_residual", True, "use residual connection for rnn cells")
    tf.flags.DEFINE_boolean('use_attention_input_feeding', True, 'Use input feeding method in attentional decoder')
    tf.flags.DEFINE_integer("maximum_iterations", 100, "maximum iterations while decoder generates outputs")
    # training parameters
    tf.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
    tf.flags.DEFINE_string("optimizer", "adam", "Optimizer: [adagrad | sgd | rmsprop | adadelta | adam], default: adam")
    tf.flags.DEFINE_boolean("use_lr_decay", True, "apply learning rate decay for each epoch")
    tf.flags.DEFINE_float("lr_decay", 0.9, "learning rate decay factor")
    tf.flags.DEFINE_float("grad_clip", 1.0, "maximal gradient norm")
    tf.flags.DEFINE_float("keep_prob", 0.6, "dropout keep probability while training")
    tf.flags.DEFINE_integer("batch_size", 128, "batch size")
    tf.flags.DEFINE_integer("epochs", 60, "train epochs")
    tf.flags.DEFINE_integer("max_to_keep", 5, "maximum trained model to be saved")
    tf.flags.DEFINE_integer("no_imprv_tolerance", 5, "no improvement tolerance")
    tf.flags.DEFINE_string("checkpoint_path", "ckpt/europarl/", "path to save model checkpoints")
    tf.flags.DEFINE_string("summary_path", "ckpt/europarl/summary/", "path to save summaries")
    return tf.flags.FLAGS.flag_values_dict()


def sentence_to_ids(sentence, source_dict, language, lower, keep_number):
    """
    :param sentence: input sentence
    :param source_dict: source dict
    :param language: source language
    :param lower: boolean, convert sentence to lowercase or not
    :param keep_number: boolean, whether to keep the number characters
    """
    if sentence is None or len(sentence) == 0:
        return None
    sentence = cleanup_sentence(sentence, language, lower, keep_number)
    tokens = sentence.split()
    ids = [source_dict.get(token, source_dict[UNK]) for token in tokens]
    return process_batch_data([ids], [[]], source_dict)


def ids_to_sentence(predict_ids, rev_target_dict, target_dict):
    """
    :param predict_ids: if GreedyDecoder -- shape = (batch_size, max_time_step, 1)
                        if BeamSearchDecoder -- shape = (batch_size, max_time_step, beam_width)
    :param rev_target_dict: reversed word dict, id-word pairs
    :param target_dict: word dict, word-id pairs
    :return: sentences
    """
    shapes = predict_ids.shape
    special_tokens = [target_dict[PAD], target_dict[GO], target_dict[EOS]]
    sentences = []
    for predict in predict_ids:
        sents = []
        for i in range(shapes[-1]):
            sent = [rev_target_dict.get(idx, UNK) for idx in predict[:, i] if idx not in special_tokens]
            sents.append(" ".join(sent))
        sentences.append(sents)
    return sentences


def play_with_model(mode):
    # build tf flags
    tf_config = create_configurations()
    # process dataset
    if not os.path.exists(tf_config["save_dir"]) or not os.listdir(tf_config["save_dir"]):
        sys.stdout.write("No preprocessed dataset found, create from {} raw data...\n"
                         .format(tf_config["dataset_name"]))
        sys.stdout.flush()
        process_europarl(tf_config)
    # create configurations
    sys.stdout.write("Load configurations...\n")
    sys.stdout.flush()
    config = Config(tf_config)
    if mode == "train":
        # prepare training dataset batches
        sys.stdout.write("Load dataset and create batches...\n")
        sys.stdout.flush()
        train_batches, test_batches = batchnize_dataset(tf_config["dataset"], config.batch_size, config.target_dict)
        # build model and start training
        sys.stdout.write("Building model...\n")
        sys.stdout.flush()
        seq2seq_model = SequenceToSequence(config, mode="train")
        seq2seq_model.train(train_batches, test_batches, epochs=config.epochs)
    elif mode == "decode":
        # build model and start training
        sys.stdout.write("Building model...\n")
        sys.stdout.flush()
        seq2seq_model = SequenceToSequence(config, mode="decode")
        seq2seq_model.restore_last_session()
        sys.stdout.write("> ")
        sys.stdout.flush()
        top_n = False  # if beam search, return all decoded results or just the first one
        source_language = "french" if tf_config["fr_to_en"] else "english"
        sentence = sys.stdin.readline()
        sent_ids = sentence_to_ids(sentence, config.source_dict, source_language, tf_config["lower"],
                                   tf_config["keep_number"])
        while sentence:
            predict_ids = seq2seq_model.inference(sent_ids)
            response = ids_to_sentence(predict_ids, config.rev_target_dict, config.target_dict)[0]  # batch_size == 1
            print(response)
            if top_n:
                print("\n".join(response))
            else:
                print(response[0])
            sys.stdout.write("> ")
            sys.stdout.flush()
            sentence = sys.stdin.readline()
            sent_ids = sentence_to_ids(sentence, config.source_dict, source_language, tf_config["lower"],
                                       tf_config["keep_number"])
    else:
        raise ValueError("ERROR: Unknown mode name {}, support modes: (train | decode)".format(mode))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, help='set task mode (train | decode).')
    args, _ = parser.parse_known_args()
    play_with_model(args.mode)
