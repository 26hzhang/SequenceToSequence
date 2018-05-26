import codecs
import re
import pickle
import ujson
import os
import string
from tqdm import tqdm
from collections import Counter
from unicodedata import normalize
from utils import UNK, GO, EOS

valid_character = re.compile('[^%s]' % re.escape(string.printable))
# alphanumeric_character = re.compile(r"[^A-Za-z_\d\- ]", re.IGNORECASE)
punct_character = str.maketrans('', '', string.punctuation)
number_character = re.compile(r"[\d]", re.IGNORECASE)


def cleanup_sentence(sent, language, lower, keep_number):
    if language == "french":
        # converts French characters to Latin equivalents
        sent = normalize("NFD", sent).encode("ascii", "ignore").decode("utf-8")
    if lower:
        sent = sent.lower()
    sent = valid_character.sub("", sent)
    sent = sent.translate(punct_character)
    if not keep_number:
        sent = number_character.sub("", sent)
    return sent


def pickle_dump(data, save_path, suffix=".pkl"):
    with codecs.open(save_path + suffix, mode='wb') as f:
        pickle.dump(data, f)


def json_dump(data, save_path, suffix=".json"):
    with codecs.open(save_path + suffix, mode='w', encoding='utf-8') as f:
        ujson.dump(data, f)


def build_vocabulary(data_pairs, en_vocab_size, fr_vocab_size):
    en_counter = Counter()
    fr_counter = Counter()
    for data in tqdm(data_pairs, desc="Build english and french vocabulary"):
        for word in data["en"]:
            en_counter[word] += 1
        for word in data["fr"]:
            fr_counter[word] += 1
    en_word_vocab = [GO, EOS, UNK] + [word for word, _ in en_counter.most_common(en_vocab_size)]
    en_word_dict = dict([(word, idx) for idx, word in enumerate(en_word_vocab)])
    fr_word_vocab = [GO, EOS, UNK] + [word for word, _ in fr_counter.most_common(fr_vocab_size)]
    fr_word_dict = dict([(word, idx) for idx, word in enumerate(fr_word_vocab)])
    return en_word_dict, fr_word_dict


def build_dataset(data_pairs, en_word_dict, fr_word_dict, fr_to_en):
    dataset = []
    for data in tqdm(data_pairs, desc="Build dataset"):
        en = [en_word_dict[word] if word in en_word_dict else en_word_dict[UNK] for word in data["en"]]
        fr = [fr_word_dict[word] if word in fr_word_dict else fr_word_dict[UNK] for word in data["fr"]]
        if fr_to_en:  # if True, build french to english translation dataset
            record = {"lu": fr, "ru": en}
        else:  # else, build english to french translation dataset
            record = {"lu": en, "ru": fr}
        dataset.append(record)
    return dataset


def create_transcript_pairs(en_fr_data, max_sent_len, min_sent_len, lower, keep_number):
    trans_pairs = []
    for en, fr in tqdm(en_fr_data, desc="Create english-french transcript pairs dataset"):
        en = cleanup_sentence(en, "english", lower, keep_number)
        fr = cleanup_sentence(fr, "french", lower, keep_number)
        en_tokens = en.split()
        fr_tokens = fr.split()
        en_len, fr_len = len(en_tokens), len(fr_tokens)
        if en_len < min_sent_len or fr_len < min_sent_len:
            continue
        if en_len > max_sent_len or fr_len > max_sent_len:
            continue
        pair = {"en": en_tokens, "fr": fr_tokens}
        trans_pairs.append(pair)
    return trans_pairs


def read_europarl_data(europarl_path):
    en_data = []
    with codecs.open(europarl_path + ".en", mode="r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Read english transcript"):
            line = line.lstrip().rstrip()
            en_data.append(line)
    fr_data = []
    with codecs.open(europarl_path + ".fr", mode="r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Read french transcript"):
            line = line.lstrip().rstrip()
            fr_data.append(line)
    assert len(en_data) == len(fr_data), "The size of english ({}) and french ({}) transcripts doesn't match".format(
        len(en_data), len(fr_data))
    dataset = []
    for en, fr in zip(en_data, fr_data):
        dataset.append((en, fr))
    return dataset


def process_europarl(tf_config):
    # read en-fr pairs
    en_fr_data = read_europarl_data(os.path.join(tf_config["raw_data_dir"], "europarl-v7.fr-en"))
    # process data
    trans_pairs = create_transcript_pairs(en_fr_data, tf_config["max_sent_len"], tf_config["min_sent_len"],
                                          tf_config["lower"], tf_config["keep_number"])
    del en_fr_data  # delete unused data to save space
    # build vocabulary
    en_word_dict, fr_word_dict = build_vocabulary(trans_pairs, tf_config["en_vocab_size"], tf_config["fr_vocab_size"])
    # build dataset
    dataset = build_dataset(trans_pairs, en_word_dict, fr_word_dict, tf_config["fr_to_en"])
    # save path
    save_path = tf_config["save_dir"]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    vocab_path = os.path.join(save_path, "vocabulary")
    dataset_path = os.path.join(save_path, "dataset")
    # write processed data to file
    if tf_config["fr_to_en"]:
        vocabulary = {"source_dict": fr_word_dict, "target_dict": en_word_dict}
    else:
        vocabulary = {"source_dict": en_word_dict, "target_dict": fr_word_dict}
    train_size = int(len(dataset) * tf_config["train_ratio"])
    json_dump({"train_set": dataset[0:train_size], "test_set": dataset[train_size:]}, dataset_path)
    json_dump(vocabulary, vocab_path)
