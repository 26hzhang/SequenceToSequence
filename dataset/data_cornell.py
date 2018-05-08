import codecs
import re
import pickle
import ujson
import os
import ast
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
    sent = duplicate_punct.sub("", sent)  # remove duplicate punctuations and keep the last one
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


def read_cornell_id_sent_pair(movie_lines):
    id_sent = {}
    with codecs.open(movie_lines, mode='r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, desc="Read cornell movie lines"):
            line = line.lstrip().rstrip().split(" +++$+++ ")
            if len(line) != 5:
                continue  # make sure current line in correct format
            id_sent[line[0]] = line[4]  # line[0] is the ID, line[4] is the utterance
    return id_sent


def read_cornell_conversation_ids(movie_conversations):
    conversation_ids = []
    with codecs.open(movie_conversations, mode='r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, desc="Read cornell movie conversations"):
            line = line.lstrip().rstrip().split(" +++$+++ ")
            if len(line) != 4:
                continue  # make sure current line in correct format
            conversation_ids.append(ast.literal_eval(line[-1]))
    return conversation_ids


def create_cornell_utter_pairs(movie_lines, movie_conversations, max_sent_len, min_sent_len, only_alphanumeric):
    id_to_sent = read_cornell_id_sent_pair(movie_lines)
    conversation_ids = read_cornell_conversation_ids(movie_conversations)
    utterances = []
    for conversation in tqdm(conversation_ids, desc="Create cornell utterance pairs"):
        if len(conversation) <= 1:  # if contains only one or no utterance, ignore it
            continue
        if len(conversation) % 2 != 0:
            conversation = conversation[:-1] + [conversation[-2], conversation[-1]]
        for i in range(0, len(conversation), 2):
            if conversation[i] not in id_to_sent or conversation[i + 1] not in id_to_sent:
                continue  # some ids corresponding to empty sentence, so ignore them
            # clean up sentences
            lu = cleanup_sentence(id_to_sent[conversation[i]].lower(), only_alphanumeric)  # left utterance
            ru = cleanup_sentence(id_to_sent[conversation[i + 1]].lower(), only_alphanumeric)  # right utterance
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


def process_cornell(tf_config):
    # input path
    movie_lines = os.path.join(tf_config["raw_data_dir"], "movie_lines.txt")
    movie_conversations = os.path.join(tf_config["raw_data_dir"], "movie_conversations.txt")
    # save path
    save_path = tf_config["save_dir"]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    meta_path = os.path.join(save_path, "metadata")
    dataset_path = os.path.join(save_path, "dataset")
    # process data
    utterances = create_cornell_utter_pairs(movie_lines, movie_conversations, tf_config["max_sent_len"],
                                            tf_config["min_sent_len"], tf_config["only_alphanumeric"])
    word_vocab, word_dict = build_vocabulary(utterances, tf_config["vocab_size"])
    dataset = build_dataset(utterances, word_dict)
    metadata = {"source_dict": {}, "target_dict": word_dict}
    # write to file
    train_size = int(len(dataset) * tf_config["train_ratio"])
    json_dump({"train_set": dataset[0:train_size], "test_set": dataset[train_size:]}, dataset_path)
    json_dump(metadata, meta_path)
