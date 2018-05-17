import codecs
import re
import pickle
import ujson
import os
from tqdm import tqdm
from collections import Counter
from nltk import word_tokenize

special_character = re.compile(r"[^A-Za-z_\d,.;!'\- ]", re.IGNORECASE)
alphanumeric_character = re.compile(r"[^A-Za-z_\d\- ]", re.IGNORECASE)
duplicate_punct = re.compile(r"[?.!,;]+(?=[?.!,;])", re.IGNORECASE)
connect_punct = re.compile(r"-+|_+", re.IGNORECASE)

PAD = "<PAD>"
UNK = "<UNK>"
GO = "<GO>"
EOS = "<EOS>"


def cleanup_sentence(sent, only_alphanumeric):
    if only_alphanumeric:
        sent = alphanumeric_character.sub("", sent)
    else:
        sent = special_character.sub("", sent)  # remove special characters
        sent = duplicate_punct.sub("", sent)  # remove duplicate punctuations and keep the last one
    sent = connect_punct.sub(" ", sent)  # replace dash and underline to space
    return sent


def pickle_dump(data, save_path, suffix=".pkl"):
    with codecs.open(save_path + suffix, mode='wb') as f:
        pickle.dump(data, f)


def json_dump(data, save_path, suffix=".json"):
    with codecs.open(save_path + suffix, mode='w', encoding='utf-8') as f:
        ujson.dump(data, f)


def build_vocabulary(utterances, max_vocab_size):
    word_counter = Counter()
    for utterance in tqdm(utterances, desc="Build vocabulary"):
        for word in utterance["lu"]:
            word_counter[word] += 1
        for word in utterance["ru"]:
            word_counter[word] += 1
    word_vocab = [PAD, GO, EOS, UNK] + [word for word, _ in word_counter.most_common(max_vocab_size)]
    word_dict = dict([(word, idx) for idx, word in enumerate(word_vocab)])
    return word_vocab, word_dict


def build_dataset(utterances, word_dict):
    dataset = []
    for utter in tqdm(utterances, desc="Build dataset"):
        lu = [word_dict[word] if word in word_dict else word_dict[UNK] for word in utter['lu']]
        ru = [word_dict[word] if word in word_dict else word_dict[UNK] for word in utter['ru']]
        record = {"lu": lu, "ru": ru}
        dataset.append(record)
    return dataset


def read_twitter_lines(twitter_path):
    data = []
    with codecs.open(twitter_path, mode='r', encoding='utf-8', errors='ignore') as f:
        index = 0
        odd_line = ""
        for line in tqdm(f, desc="Read twitter chat utterances:"):
            index += 1
            if index % 2 == 0:
                even_line = line.lstrip().rstrip()
                data.append((odd_line, even_line))
            else:
                odd_line = line.lstrip().rstrip()
    return data


def create_twitter_utter_pairs(dataset, max_sent_len, min_sent_len, only_alphanumeric):
    utterances = []
    for odd_line, even_line in tqdm(dataset, desc="Create twitter utterance pairs"):
        # lowercase and cleanup sentence
        lu = cleanup_sentence(odd_line.lower(), only_alphanumeric)
        ru = cleanup_sentence(even_line.lower(), only_alphanumeric)
        # tokenizing and filtering
        lu_words, ru_words = word_tokenize(lu), word_tokenize(ru)
        lu_length, ru_length = len(lu_words), len(ru_words)
        if lu_length < min_sent_len or ru_length < min_sent_len:
            continue
        lu_words = lu_words[:max_sent_len] if lu_length > max_sent_len else lu_words
        ru_words = ru_words[:max_sent_len] if ru_length > max_sent_len else ru_words
        utter = {"lu": lu_words, "ru": ru_words}
        utterances.append(utter)
    return utterances


def process_twitter(tf_config):
    twitter_line_pairs = read_twitter_lines(os.path.join(tf_config["raw_data_dir"], "twitter_en.txt"))
    # save path
    save_path = tf_config["save_dir"]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    vocab_path = os.path.join(save_path, "vocabulary")
    dataset_path = os.path.join(save_path, "dataset")
    # process data
    utterances = create_twitter_utter_pairs(twitter_line_pairs, tf_config["max_sent_len"], tf_config["min_sent_len"],
                                            tf_config["only_alphanumeric"])
    word_vocab, word_dict = build_vocabulary(utterances, tf_config["vocab_size"])
    dataset = build_dataset(utterances, word_dict)
    vocabulary = {"source_dict": {}, "target_dict": word_dict}
    # write to file
    train_size = int(len(dataset) * tf_config["train_ratio"])
    json_dump({"train_set": dataset[0:train_size], "test_set": dataset[train_size:]}, dataset_path)
    json_dump(vocabulary, vocab_path)
