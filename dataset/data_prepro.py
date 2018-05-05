import codecs
import os
import ast
import re
import pickle
import ujson
from tqdm import tqdm
from nltk import word_tokenize
from collections import Counter

special_character = re.compile(r"[^A-Za-z_\d,.;!?'\- ]", re.IGNORECASE)
duplicate_punct = re.compile(r"[?.!,;]+(?=[?.!,;])", re.IGNORECASE)
connect_punct = re.compile(r"-+|_+", re.IGNORECASE)

MAX_SENT_LENGTH = 40
MIN_SENT_LENGTH = 2
MAX_VOCAB_SIZE = 10000

PAD = "<PAD>"
UNK = "<UNK>"
GO = "<GO>"
EOS = "<EOS>"


def cleanup_sentence(sent):
    sent = special_character.sub("", sent)  # remove special characters
    sent = connect_punct.sub(" ", sent)  # replace dash and underline to space
    sent = duplicate_punct.sub("", sent)  # remove duplicate punctuations and keep the last one
    return sent


def pickle_dump(data, save_path, suffix=".pkl"):
    with codecs.open(save_path + suffix, mode='wb') as f:
        pickle.dump(data, f)


def json_dump(data, save_path, suffix=".json"):
    with codecs.open(save_path + suffix, mode='w', encoding='utf-8') as f:
        ujson.dump(data, f)


def read_id_sent_pair(movie_lines):
    id_sent = {}
    with codecs.open(movie_lines, mode='r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, desc="Read movie lines"):
            line = line.lstrip().rstrip().split(" +++$+++ ")
            if len(line) != 5:
                continue  # make sure current line in correct format
            id_sent[line[0]] = line[4]  # line[0] is the ID, line[4] is the utterance
    return id_sent


def read_conversation_ids(movie_conversations):
    conversation_ids = []
    with codecs.open(movie_conversations, mode='r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, desc="Read movie conversations"):
            line = line.lstrip().rstrip().split(" +++$+++ ")
            if len(line) != 4:
                continue  # make sure current line in correct format
            conversation_ids.append(ast.literal_eval(line[-1]))
    return conversation_ids


def create_utterance_pair(movie_lines, movie_conversations):
    id_to_sent = read_id_sent_pair(movie_lines)
    conversation_ids = read_conversation_ids(movie_conversations)
    utterances = []
    for conversation in tqdm(conversation_ids, desc="Create utterance pairs"):
        if len(conversation) <= 1:  # if contains only one or no utterance, ignore it
            continue
        if len(conversation) % 2 != 0:
            conversation = conversation[:-1] + [conversation[-2], conversation[-1]]
        for i in range(0, len(conversation), 2):
            if conversation[i] not in id_to_sent or conversation[i + 1] not in id_to_sent:
                continue  # some ids corresponding to empty sentence, so ignore them
            # clean up sentences
            lu = cleanup_sentence(id_to_sent[conversation[i]].lower())  # left utterance
            ru = cleanup_sentence(id_to_sent[conversation[i + 1]].lower())  # right utterance
            # tokenizing and filtering
            lu_words, ru_words = word_tokenize(lu), word_tokenize(ru)
            lu_length, ru_length = len(lu_words), len(ru_words)
            if lu_length < MIN_SENT_LENGTH or ru_length < MIN_SENT_LENGTH:
                continue
            lu_words = lu_words[:MAX_SENT_LENGTH] if lu_length > MAX_SENT_LENGTH else lu_words
            ru_words = ru_words[:MAX_SENT_LENGTH] if ru_length > MAX_SENT_LENGTH else ru_words
            utter = {"lu": lu_words, "ru": ru_words}
            utterances.append(utter)
    return utterances


def build_vocabulary(utterances):
    word_counter = Counter()
    for utterance in tqdm(utterances, desc="Build vocabulary"):
        for word in utterance["lu"]:
            word_counter[word] += 1
        for word in utterance["ru"]:
            word_counter[word] += 1
    word_vocab = [PAD, GO, EOS, UNK] + [word for word, _ in word_counter.most_common(MAX_VOCAB_SIZE)]
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


def process_datasets(raw_data_dir, save_dir, train_ratio=0.95):
    # input path
    movie_lines = os.path.join(raw_data_dir, "movie_lines.txt")
    movie_conversations = os.path.join(raw_data_dir, "movie_conversations.txt")
    # save path
    meta_path = os.path.join(save_dir, "metadata")
    dataset_path = os.path.join(save_dir, "dataset")
    # process data
    utterances = create_utterance_pair(movie_lines, movie_conversations)
    word_vocab, word_dict = build_vocabulary(utterances)
    dataset = build_dataset(utterances, word_dict)
    metadata = {"vocab": word_vocab, "dict": word_dict}
    # write to file
    train_size = int(len(dataset) * train_ratio)
    json_dump({"train_set": dataset[0:train_size], "test_set": dataset[train_size:]}, dataset_path)
    json_dump(metadata, meta_path)


def main():  # for testing
    # input path
    movie_lines = os.path.join("raw", "movie_lines.txt")
    movie_conversations = os.path.join("raw", "movie_conversations.txt")
    # output path
    meta_path = os.path.join("data", "metadata")
    dataset_path = os.path.join("data", "dataset")
    # process data
    utterances = create_utterance_pair(movie_lines, movie_conversations)
    word_vocab, word_dict = build_vocabulary(utterances)
    dataset = build_dataset(utterances, word_dict)
    metadata = {"dict": word_dict, "vocab": word_vocab}
    # write to file
    json_dump(dataset, dataset_path)
    json_dump(metadata, meta_path)


if __name__ == "__main__":
    main()
