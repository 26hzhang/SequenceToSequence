import ujson
import pickle
import codecs
import random
from dataset.data_prepro import GO, EOS, PAD


def load_data(filename):
    if filename.endswith(".json"):
        with codecs.open(filename, mode='r', encoding='utf-8', errors='ignore') as f:
            data = ujson.load(f)
        return data
    elif filename.endswith(".pkl"):
        with codecs.open(filename, mode='rb') as f:
            data = pickle.load(f)
        return data
    else:
        raise ValueError("ERROR: Unknown file extension, only support `.json` and `.pkl` formats!!!")


def process_batch_data(batch_lu, batch_ru, word_dict):
    batch_lu_len = [len(lu) for lu in batch_lu]
    batch_ru_len = [len(ru) for ru in batch_ru]
    b_lu, b_ru = [], []
    max_lu_len = max(batch_lu_len)
    max_ru_len = max(batch_ru_len)
    for lu, ru in zip(batch_lu, batch_ru):
        lu = [word_dict[PAD]] * (max_lu_len - len(lu)) + list(reversed(lu))  # reverse and PAD left utterance
        ru = ru + [word_dict[PAD]] * (max_ru_len - len(ru))
        b_lu.append(lu)
        b_ru.append(ru)
    return {"enc_input": b_lu, "enc_seq_len": batch_lu_len, "dec_input": b_ru, "dec_seq_len": batch_ru_len}


def dataset_batch_iter(dataset, batch_size, word_dict, shuffle=True):
    if shuffle:
        random.shuffle(dataset)
    batch_lu, batch_ru = [], []
    for record in dataset:
        batch_lu.append(record["lu"])
        batch_ru.append(record["ru"])
        if len(batch_lu) == batch_size:
            yield process_batch_data(batch_lu, batch_ru, word_dict)
            batch_lu, batch_ru = [], []
    if len(batch_lu) > 0:
        yield process_batch_data(batch_lu, batch_ru, word_dict)


def batchnize_dataset(dataset, batch_size, word_dict):
    batches = []
    for batch in dataset_batch_iter(dataset, batch_size, word_dict, shuffle=True):
        batches.append(batch)
    return batches
