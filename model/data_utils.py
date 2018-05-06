import ujson
import pickle
import codecs
import random
from tqdm import tqdm
from nltk import word_tokenize
from dataset.data_prepro import cleanup_sentence
from dataset.data_prepro import GO, EOS, PAD, UNK


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
    max_ru_len = max([len(ru) for ru in batch_ru])
    b_lu, b_ru_in, b_ru_out = [], [], []
    for lu, ru in zip(batch_lu, batch_ru):
        # reverse encoder input and add PAD at the begin
        lu = [word_dict[PAD]] * (max_lu_len - len(lu)) + list(reversed(lu))  # reverse and PAD encoder input
        # add GO at the begin and add PAD at the end
        ru_in = [word_dict[GO]] + ru + [word_dict[PAD]] * (max_ru_len - len(ru))  # add GO for decoder input
        # add EOS at the end and add PAD at the end
        ru_out = ru + [word_dict[EOS]] + [word_dict[PAD]] * (max_ru_len - len(ru))  # add EOS for decoder output
        b_lu.append(lu)
        b_ru_in.append(ru_in)
        b_ru_out.append(ru_out)
    batch_ru_len = [len(ru) for ru in b_ru_in]
    return {"enc_input": b_lu, "enc_seq_len": batch_lu_len, "dec_input": b_ru_in, "dec_output": b_ru_out,
            "dec_seq_len": batch_ru_len, "batch_size": len(b_lu)}


def dataset_batch_iter(dataset, batch_size, word_dict, shuffle=False):
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


def batchnize_dataset(filename, batch_size, word_dict):
    dataset = load_data(filename)
    train_batches = []
    for batch in tqdm(dataset_batch_iter(dataset["train_set"], batch_size, word_dict, shuffle=True),
                      desc="Prepare train batches"):
        train_batches.append(batch)
    test_batches = []
    for batch in tqdm(dataset_batch_iter(dataset["test_set"], batch_size, word_dict), desc="Prepare test batches"):
        test_batches.append(batch)
    return train_batches, test_batches


def sentence_to_ids(sentence, word_dict):
    if sentence is None or len(sentence) == 0:
        return None
    sentence = cleanup_sentence(sentence)
    tokens = word_tokenize(sentence)
    ids = [word_dict.get(token, word_dict[UNK]) for token in tokens]
    return process_batch_data([ids], [[]], word_dict)  # the decode data is not used


def ids_to_sentence(predict_ids, rev_word_dict, beam_size):
    """
    :param predict_ids: if GreedyDecoder -- shape = (batch_size, max_time_step, 1)
                        if BeamSearchDecoder -- shape = (batch_size, max_time_step, beam_width)
    :param rev_word_dict: reversed word dict, id-word pairs
    :param beam_size: beam size
    :return: sentences
    """
    sentences = []
    for predict in predict_ids:
        for i in range(beam_size):
            pred_list = predict[:, :, i]
            sent = [rev_word_dict.get(idx, UNK) for idx in pred_list[0]]
            sentences.append(" ".join(sent))
    return sentences
