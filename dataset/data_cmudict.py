import codecs
import pickle
import ujson
import os
from tqdm import tqdm
from collections import Counter

PAD = "<PAD>"
UNK = "<UNK>"
GO = "<GO>"
EOS = "<EOS>"


def pickle_dump(data, save_path, suffix=".pkl"):
    with codecs.open(save_path + suffix, mode='wb') as f:
        pickle.dump(data, f)


def json_dump(data, save_path, suffix=".json"):
    with codecs.open(save_path + suffix, mode='w', encoding='utf-8') as f:
        ujson.dump(data, f)


def build_cmudict_vocabulary(word_phoneme_pairs):
    # characters = "_abcdefghijklmnopqrstuvwxyz"
    char_counter, phoneme_counter = Counter(), Counter()
    for pair in tqdm(word_phoneme_pairs, desc="Build vocabulary"):
        for char in pair["chars"]:
            char_counter[char] += 1
        for phoneme in pair["phoneme"]:
            phoneme_counter[phoneme] += 1
    char_vocab = [PAD, GO, EOS, UNK] + [char for char, _ in char_counter.most_common()]
    char_dict = dict([(char, idx) for idx, char in enumerate(char_vocab)])
    phoneme_vocab = [PAD, UNK] + [phoneme for phoneme, _ in phoneme_counter.most_common()]
    phoneme_dict = dict([(v, i) for i, v in enumerate(phoneme_vocab)])
    return char_dict, phoneme_dict


def build_cmudict_dataset(word_phoneme_pairs, char_dict, phoneme_dict):
    dataset = []
    for data in tqdm(word_phoneme_pairs, desc="Build dataset"):
        lu = [phoneme_dict[phoneme] for phoneme in data["phoneme"]]
        ru = [char_dict[char] for char in data["chars"]]
        record = {"lu": lu, "ru": ru}
        dataset.append(record)
    return dataset


def read_cmudict_lines(cmudict_path, min_size, max_size):
    start_line = 126
    end_line = 133905
    with codecs.open(cmudict_path, mode='r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()[start_line: end_line]
    word_phoneme_pairs = []
    for line in tqdm(lines, desc="Read and filter cmudict lines"):
        word, phoneme = line.lstrip().rstrip().split("  ")
        chars = list(word.lower())
        phonemes = phoneme.split(" ")
        if not word.isalpha():
            continue  # if only_alpha is True, then all the word contains non-alpha characters will be ignored
        if len(chars) < min_size or len(phonemes) < min_size:
            continue
        if len(chars) > max_size or len(phonemes) > max_size:
            continue
        word_phoneme_pairs.append({"chars": chars, "phoneme": phonemes})
    return word_phoneme_pairs


def process_cmudict(tf_config):
    word_phoneme_pairs = read_cmudict_lines(os.path.join(tf_config["raw_data_dir"], "cmudict-0.7b"),
                                            tf_config["min_size"], tf_config["max_size"])
    # save path
    save_path = tf_config["save_dir"]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    meta_path = os.path.join(save_path, "metadata")
    dataset_path = os.path.join(save_path, "dataset")
    # process data
    char_dict, phoneme_dict = build_cmudict_vocabulary(word_phoneme_pairs)
    dataset = build_cmudict_dataset(word_phoneme_pairs, char_dict, phoneme_dict)
    meta_data = {"source_dict": phoneme_dict, "target_dict": char_dict}
    # write to file
    train_size = int(len(dataset) * tf_config["train_ratio"])
    json_dump({"train_set": dataset[0:train_size], "test_set": dataset[train_size:]}, dataset_path)
    json_dump(meta_data, meta_path)
