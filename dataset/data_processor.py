import codecs
import re
import pickle
import ujson
import os
import ast
import sys
from tqdm import tqdm
from collections import Counter
from nltk import word_tokenize

special_character = re.compile(r"[^A-Za-z_\d,.;!?'\- ]", re.IGNORECASE)
duplicate_punct = re.compile(r"[?.!,;]+(?=[?.!,;])", re.IGNORECASE)
connect_punct = re.compile(r"-+|_+", re.IGNORECASE)

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


def build_cmudict_vocabulary(word_phoneme_pairs):
    # characters = "_abcdefghijklmnopqrstuvwxyz"
    # char_dict = dict({v: i for i, v in enumerate(characters)})
    char_counter, phoneme_counter = Counter(), Counter()
    for pair in tqdm(word_phoneme_pairs, desc="Build vocabulary"):
        for char in pair["chars"]:
            char_counter[char] += 1
        for phoneme in pair["phoneme"]:
            phoneme_counter[phoneme] += 1
    char_vocab = [char for char, _ in char_counter.most_common()]
    char_dict = dict([(char, idx) for idx, char in enumerate(char_vocab)])
    phoneme_vocab = [phoneme for phoneme, _ in phoneme_counter.most_common()]
    phoneme_dict = dict([(v, i) for i, v in enumerate(phoneme_vocab)])
    return char_dict, phoneme_dict


def build_dataset(utterances, word_dict):
    dataset = []
    for utter in tqdm(utterances, desc="Build dataset"):
        lu = [word_dict[word] if word in word_dict else word_dict[UNK] for word in utter['lu']]
        ru = [word_dict[word] if word in word_dict else word_dict[UNK] for word in utter['ru']]
        record = {"lu": lu, "ru": ru}
        dataset.append(record)
    return dataset


def build_cmudict_dataset(word_phoneme_pairs, char_dict, phoneme_dict):
    dataset = []
    for data in tqdm(word_phoneme_pairs, desc="Build dataset"):
        lu = [phoneme_dict[phoneme] for phoneme in data["phoneme"]]
        ru = [char_dict[char] for char in data["chars"]]
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


def read_cmudict_lines(cmudict_path, min_size=5, max_size=20):
    start_line = 126
    end_line = 133905
    with codecs.open(cmudict_path, mode='r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()[start_line: end_line]
    word_phoneme_pairs = []
    for line in tqdm(lines, desc="Read and filter cmudict lines"):
        word, phoneme = line.lstrip().rstrip().split("  ")
        chars = list(word.lower())
        phonemes = phoneme.split(" ")
        if len(chars) < min_size or len(phonemes) < min_size:
            continue
        if len(chars) > max_size or len(phonemes) > max_size:
            continue
        word_phoneme_pairs.append({"chars": chars, "phoneme": phoneme})
    return word_phoneme_pairs


def create_cornell_utter_pairs(movie_lines, movie_conversations, max_sent_len, min_sent_len):
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
            lu = cleanup_sentence(id_to_sent[conversation[i]].lower())  # left utterance
            ru = cleanup_sentence(id_to_sent[conversation[i + 1]].lower())  # right utterance
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


def create_twitter_utter_pairs(dataset, max_sent_len, min_sent_len):
    utterances = []
    for odd_line, even_line in tqdm(dataset, desc="Create twitter utterance pairs"):
        # lowercase and cleanup sentence
        lu = cleanup_sentence(odd_line.lower())
        ru = cleanup_sentence(even_line.lower())
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


def process_cornell(raw_data_dir, save_dir, train_ratio=0.95, max_vocab_size=10000, max_sent_len=40, min_sent_len=2):
    # input path
    movie_lines = os.path.join(raw_data_dir, "cornell", "movie_lines.txt")
    movie_conversations = os.path.join(raw_data_dir, "cornell", "movie_conversations.txt")
    # save path
    save_path = os.path.join(save_dir, "cornell")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    meta_path = os.path.join(save_path, "metadata")
    dataset_path = os.path.join(save_path, "dataset")
    # process data
    utterances = create_cornell_utter_pairs(movie_lines, movie_conversations, max_sent_len, min_sent_len)
    word_vocab, word_dict = build_vocabulary(utterances, max_vocab_size)
    dataset = build_dataset(utterances, word_dict)
    metadata = {"vocab": word_vocab, "dict": word_dict}
    # write to file
    train_size = int(len(dataset) * train_ratio)
    json_dump({"train_set": dataset[0:train_size], "test_set": dataset[train_size:]}, dataset_path)
    json_dump(metadata, meta_path)


def process_twitter(raw_data_dir, save_dir, train_ratio=0.95, min_vocab_size=10000, max_sent_len=40, min_sent_len=2):
    twitter_line_pairs = read_twitter_lines(os.path.join(raw_data_dir, "twitter", "twitter_en.txt"))
    # save path
    save_path = os.path.join(save_dir, "twitter")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    meta_path = os.path.join(save_path, "metadata")
    dataset_path = os.path.join(save_path, "dataset")
    # process data
    utterances = create_twitter_utter_pairs(twitter_line_pairs, max_sent_len, min_sent_len)
    word_vocab, word_dict = build_vocabulary(utterances, min_vocab_size)
    dataset = build_dataset(utterances, word_dict)
    metadata = {"vocab": word_vocab, "dict": word_dict}
    # write to file
    train_size = int(len(dataset) * train_ratio)
    json_dump({"train_set": dataset[0:train_size], "test_set": dataset[train_size:]}, dataset_path)
    json_dump(metadata, meta_path)


def process_cmudict(raw_data_dir, save_dir, train_ratio=0.95):
    word_phoneme_pairs = read_cmudict_lines(os.path.join(raw_data_dir, "cmudict", "cmudict-0.7b"))
    # save path
    save_path = os.path.join(save_dir, "cmudict")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    meta_path = os.path.join(save_path, "metadata")
    dataset_path = os.path.join(save_path, "dataset")
    # process data
    char_dict, phoneme_dict = build_cmudict_vocabulary(word_phoneme_pairs)
    dataset = build_cmudict_dataset(word_phoneme_pairs, char_dict, phoneme_dict)
    meta_data = {"char_dict": char_dict, "phoneme_dict": phoneme_dict}
    # write to file
    train_size = int(len(dataset) * train_ratio)
    json_dump({"train_set": dataset[0:train_size], "test_set": dataset[train_size:]}, dataset_path)
    json_dump(meta_data, meta_path)


def create_dataset(raw_data_dir, save_dir, dataset_name):
    if dataset_name == "cornell":
        sys.stdout.write("Process Cornell Movie Dialogs datasets:\n")
        sys.stdout.flush()
        process_cornell(raw_data_dir, save_dir)
        sys.stdout.write("done...\n")
        sys.stdout.flush()
    elif dataset_name == "twitter":
        sys.stdout.write("Process Twitter Chat datasets:\n")
        sys.stdout.flush()
        process_twitter(raw_data_dir, save_dir)
        sys.stdout.write("done...\n")
        sys.stdout.flush()
    elif dataset_name == "cmudict":
        sys.stdout.write("Process CMU Pronouncing Dictionary datasets:\n")
        sys.stdout.flush()
        process_cmudict(raw_data_dir, save_dir)
        sys.stdout.write("done...\n")
        sys.stdout.flush()
    else:
        raise ValueError("ERROR: Unknown dataset, please select a dataset in {\"cornell\", \"twitter\", \"cmudict\"}")
