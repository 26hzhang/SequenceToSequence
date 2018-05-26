import ujson
import pickle
import codecs
import random
from tqdm import tqdm

UNK = "<UNK>"
GO = "<GO>"
EOS = "<EOS>"  # also act as PAD


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
    max_lu_len = max(batch_lu_len)
    batch_ru_len = [len(ru) for ru in batch_ru]
    max_ru_len = max(batch_ru_len)
    b_lu, b_ru_in, b_ru_out = [], [], []
    for lu, ru in zip(batch_lu, batch_ru):
        # reverse encoder input and add PAD at the end
        # lu = list(reversed(lu)) + [word_dict[PAD]] * (max_lu_len - len(lu))  # reverse and PAD encoder input
        lu = lu + [word_dict[EOS]] * (max_lu_len - len(lu))  # reverse and PAD encoder input
        # add GO at the begin and add PAD at the end for decoder input
        ru_in = [word_dict[GO]] + ru + [word_dict[EOS]] * (max_ru_len - len(ru))  # add GO for decoder input
        # add PAD and EOS at the end for decoder output
        ru_out = ru + [word_dict[EOS]] + [word_dict[EOS]] * (max_ru_len - len(ru))  # add EOS for decoder output
        b_lu.append(lu)
        b_ru_in.append(ru_in)
        b_ru_out.append(ru_out)
    batch_ru_len = [x + 1 for x in batch_ru_len]  # add 1 for each because of GO and EOS token
    return {"source_in": b_lu, "source_len": batch_lu_len, "target_in": b_ru_in, "target_out": b_ru_out,
            "target_len": batch_ru_len, "batch_size": len(b_lu)}


def dataset_batch_iter(dataset, batch_size, word_dict, shuffle):
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


def batchnize_dataset(filename, batch_size, word_dict, shuffle=True):
    dataset = load_data(filename)
    train_batches = []
    for batch in tqdm(dataset_batch_iter(dataset["train_set"], batch_size, word_dict, shuffle=shuffle),
                      desc="Prepare train batches"):
        train_batches.append(batch)
    test_batches = []
    for batch in tqdm(dataset_batch_iter(dataset["test_set"], batch_size, word_dict, shuffle=shuffle),
                      desc="Prepare test batches"):
        test_batches.append(batch)
    return train_batches, test_batches
