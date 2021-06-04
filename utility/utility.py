import argparse
import json
import pickle
import random
import re
import shutil
import math
from copy import deepcopy as cp

import numpy as np
import torch
import torch.nn.functional as F


# IO
def loadFromJson(filename):
    f = open(filename, 'r', encoding='utf-8')
    data = json.load(f, strict=False)
    f.close()
    return data


def saveToJson(filename, data):
    f = open(filename, 'w', encoding='utf-8')
    json.dump(data, f, indent=4)
    f.close()
    return True


def saveToPKL(filename, data):
    with open(filename, 'wb')as f:
        pickle.dump(data, f)
    return


def loadFromPKL(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def writeFile(filename, massage):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(massage)
    return True


# Initializing
def zero_weights(n_in, n_out=None):
    if (n_out == None):
        W = np.zeros(n_in)
    else:
        W = np.zeros(n_in, n_out)
    return W.astype('float32')


def orthogonal_weights(n_dim):
    W = np.random.randn(n_dim, n_dim)
    u, _, _ = np.linalg.svd(W)
    return u.astype('float32')


def random_weights(n_in, n_out, scale=None):
    if scale is None:
        scale = np.sqrt(2.0 / (n_in + n_out))
    W = scale * np.random.randn(n_in, n_out)
    return W.astype('float32')


def remove_digits(parse):
    return re.sub(r'\d', '#', parse)


def save_check_point(state, is_best, path='.model', fileName='latest.pth.tar'):
    torch.save(state, path + '/' + fileName)
    if is_best:
        shutil.copyfile(path + '/' + fileName, path + '/model_best.pth.tar')
        shutil.copyfile(path + '/' + fileName, path + '/model_best_epoch_' + str(state['epoch']) + '.pth.tar')


def RougeTrick(parse):
    parse = re.sub(r'#', 'T', parse)
    parse = re.sub(r'T-', 'TD', parse)
    parse = re.sub(r'-T', 'DT', parse)
    parse = re.sub(r'TX.', 'TB', parse)
    parse = re.sub(r'.T', 'BT', parse)
    parse = re.sub(r'<unk>', 'UNK', parse)

    return parse


def from_dict(json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    config = argparse.Namespace()
    for key, value in json_object.items():
        config.__dict__[key] = value
    return config


def from_json_file(json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    with open(json_file, "r", encoding='utf-8') as reader:
        text = reader.read()
    return from_dict(json.loads(text))


def Index2Word(Index, Vocab):
    return Vocab.i2w[Index]


def Word2Index(Word, Vocab):
    if (not Word in Vocab.w2i):
        Word = '<unk>'
    return Vocab.w2i[Word]


def Sentence2ListOfWord(sentence):
    listOfWord = sentence.split()
    return listOfWord


def ListOfWord2ListOfIndex(listOfWord, Vocab):
    listOfIndex = []
    for w in listOfWord:
        listOfIndex.append(Word2Index(w, Vocab))
    return listOfIndex


def Sentence2ListOfIndex(sentence, Vocab):
    return ListOfWord2ListOfIndex(Sentence2ListOfWord(sentence), Vocab)


def maskDropout_col(x, r, mask):
    mc = x.data.new(1, x.size(1)).bernoulli_(r)
    return x * (1 - mc * mask)


def prefixLMMask(n, m):
    mask = torch.zeros(n + m, n + m)
    mask[:n, :n] = torch.ones(n, n)
    mask[n:, :n] = torch.ones(m, n)
    mask[n:, n:] = torch.tril(torch.ones(m, m))
    return mask

def text2textMask(n, m):
    mask = torch.zeros(n + m, n + m)
    mask[:n, :n] = torch.ones(n, n)
    mask[n:, :n] = torch.ones(m, n)
    mask[n:, n:] = torch.ones(m, m)
    return mask

def constant_Corruption(config, tick, switch):
    if switch:
        return config.corruption_target
    else:
        return config.corruption_source

def linear_Corruption(config, tick, switch):
    if switch:
        period = config.corruption_target_period
        start = config.corruption_target_start
        end = config.corruption_target_end
    else:
        period = config.corruption_source_period
        start = config.corruption_source_start
        end = config.corruption_source_end

    phi = tick / period
    phi -= math.floor(phi)
    return start + (end - start) * phi


def triangle_Corruption(config, tick, switch):
    if switch:
        period = config.corruption_target_period
        start = config.corruption_target_start
        end = config.corruption_target_end
    else:
        period = config.corruption_source_period
        start = config.corruption_source_start
        end = config.corruption_source_end

    phi = tick / period
    return (-math.cos(phi) + 1) / 2 * (end - start) + start


corruptionFunc = {
    "linear": linear_Corruption,
    "constant": constant_Corruption,
    "triangle": triangle_Corruption
}

def caclProb(config, tick, switch):
    if tick < 0:
        if not switch:
            return 0.0
    return corruptionFunc[config.corruption_schedule](config, tick, switch)


def maskSeq(seq, tick, config, switch = True):
    length = len(seq)
    prob = caclProb(config, tick, switch)
    seq_input = []
    seq_label = []
    for i in range(length):
        rrd = random.random()
        if (rrd < prob):
            seq_label.append(seq[i])
            rd = random.random()
            if rd < config.corruption_mask_rates:
                seq_input.append(config.MASK)
            elif rd < config.corruption_mask_rates + config.corruption_random_rates:
                seq_input.append(random.choice(list(range(config.n_vocab))))
            else:
                seq_input.append(seq[i])
        else:
            seq_label.append(config.PAD)
            seq_input.append(seq[i])
    return seq_input, seq_label

def prepare_data(srcs, tgts, tick, config):
    ls = [len(src) + len(tgt) + 3 for src, tgt in zip(srcs, tgts)]
    l_max = max(ls)

    padded_inputs = []
    padded_positions = []
    padded_token_types = []

    padded_labels = []
    padded_masks = []

    for src, tgt, l in zip(srcs, tgts, ls):
        src_ = [config.CLS] + src + [config.SEP]
        tgt_ = tgt + [config.SEP]


        # Mask Both Sides
        new_src_input, new_src_label = maskSeq(src_, tick, config, False)
        new_tgt_input, new_tgt_label = maskSeq(tgt_, tick, config, True)

        # Cast to Torch Types

        input_i = torch.LongTensor(new_src_input + new_tgt_input)
        position_i = torch.cat([torch.arange(len(src_)), torch.arange(1, len(tgt_) + 1)]).long()
        token_type_i = torch.LongTensor([0] * (len(src_)) + [1] * len(tgt_))
        label_i = torch.LongTensor(new_src_label + new_tgt_label)

        mask_i = torch.ones(len(src_) + len(tgt_), len(src_) + len(tgt_))

        # Padding
        padded_input_i = F.pad(input_i, (0, l_max - l), "constant", config.PAD)
        padded_position_i = F.pad(position_i, (0, l_max - l), "constant", 0)
        padded_token_type_i = F.pad(token_type_i, (0, l_max - l), "constant", 0)

        padded_label_i = F.pad(label_i, (0, l_max - l), "constant", config.PAD)
        padded_mask_i = F.pad(mask_i, (0, l_max - l, 0, l_max - l), "constant", 0.0)

        # Append to list

        padded_inputs.append(padded_input_i)
        padded_positions.append(padded_position_i)
        padded_token_types.append(padded_token_type_i)

        padded_labels.append(padded_label_i)
        padded_masks.append(padded_mask_i)

    inputs, positions, token_types, labels, masks = torch.stack(padded_inputs), \
                                                    torch.stack(padded_positions), torch.stack(padded_token_types), \
                                                    torch.stack(padded_labels), torch.stack(padded_masks)

    #if config.parallel:
    inputs = inputs.cuda(config.device)
    positions = positions.cuda(config.device)
    token_types = token_types.cuda(config.device)
    labels = labels.cuda(config.device)
    masks = masks.cuda(config.device)

    return [inputs, positions, token_types, labels, masks]

def prepare_data_cls(srcs, tgts, labels, config):
    l_alls = [len(src) + len(tgt) + 3 for src, tgt in zip(srcs, tgts)]
    l_srcs = [len(src) + 2 for src in srcs]
    l_tgts = [len(tgt) + 2 for tgt in tgts]
    l_max_all = max(l_alls)
    l_max_src = max(l_srcs)
    l_max_tgt = max(l_tgts)

    padded_src_inputs = []
    padded_src_positions = []
    padded_src_token_types = []
    padded_src_masks = []

    padded_tgt_inputs = []
    padded_tgt_positions = []
    padded_tgt_token_types = []
    padded_tgt_masks = []

    padded_inputs = []
    padded_positions = []
    padded_token_types = []
    padded_masks = []

    for src, tgt, l_all, l_src, l_tgt in zip(srcs, tgts, l_alls, l_srcs, l_tgts):
        _src_ = [config.CLS] + src + [config.SEP]
        tgt_ = tgt + [config.SEP]
        _tgt_ = [config.CLS] + tgt + [config.SEP]

        src_input_i = torch.LongTensor(_src_)
        src_position_i = torch.arange(len(_src_)).long()
        src_token_type_i = torch.LongTensor([0] * len(_src_))
        src_mask_i = torch.ones(len(_src_), len(_src_))

        tgt_input_i = torch.LongTensor(_tgt_)
        tgt_position_i = torch.arange(len(_tgt_)).long()
        tgt_token_type_i = torch.LongTensor([1] * len(_tgt_))
        tgt_mask_i = torch.ones(len(_tgt_), len(_tgt_))

        input_i = torch.LongTensor(_src_ + tgt_)
        position_i = torch.cat([torch.arange(len(_src_)), torch.arange(1, len(tgt_) + 1)]).long()
        token_type_i = torch.LongTensor([0] * (len(_src_)) + [1] * len(tgt_))
        mask_i = torch.ones(len(_src_) + len(tgt_), len(_src_) + len(tgt_))

        padded_src_input_i = F.pad(src_input_i, (0, l_max_src - l_src), "constant", config.PAD)
        padded_src_position_i = F.pad(src_position_i, (0, l_max_src - l_src), "constant", 0)
        padded_src_token_type_i = F.pad(src_token_type_i, (0, l_max_src - l_src), "constant", 0)
        padded_src_mask_i = F.pad(src_mask_i, (0, l_max_src - l_src, 0, l_max_src - l_src), "constant", 0.0)

        padded_tgt_input_i = F.pad(tgt_input_i, (0, l_max_tgt - l_tgt), "constant", config.PAD)
        padded_tgt_position_i = F.pad(tgt_position_i, (0, l_max_tgt - l_tgt), "constant", 0)
        padded_tgt_token_type_i = F.pad(tgt_token_type_i, (0, l_max_tgt - l_tgt), "constant", 0)
        padded_tgt_mask_i = F.pad(tgt_mask_i, (0, l_max_tgt - l_tgt, 0, l_max_tgt - l_tgt), "constant", 0.0)

        padded_input_i = F.pad(input_i, (0, l_max_all - l_all), "constant", config.PAD)
        padded_position_i = F.pad(position_i, (0, l_max_all - l_all), "constant", 0)
        padded_token_type_i = F.pad(token_type_i, (0, l_max_all - l_all), "constant", 0)
        padded_mask_i = F.pad(mask_i, (0, l_max_all - l_all, 0, l_max_all - l_all), "constant", 0.0)

        padded_src_inputs.append(padded_src_input_i)
        padded_src_positions.append(padded_src_position_i)
        padded_src_token_types.append(padded_src_token_type_i)
        padded_src_masks.append(padded_src_mask_i)

        padded_tgt_inputs.append(padded_tgt_input_i)
        padded_tgt_positions.append(padded_tgt_position_i)
        padded_tgt_token_types.append(padded_tgt_token_type_i)
        padded_tgt_masks.append(padded_tgt_mask_i)

        padded_inputs.append(padded_input_i)
        padded_positions.append(padded_position_i)
        padded_token_types.append(padded_token_type_i)
        padded_masks.append(padded_mask_i)

    labels = torch.FloatTensor(labels)
    src_inputs, src_positions, src_token_types, src_masks = torch.stack(padded_src_inputs), \
                                                            torch.stack(padded_src_positions), \
                                                            torch.stack(padded_src_token_types), \
                                                            torch.stack(padded_src_masks)

    tgt_inputs, tgt_positions, tgt_token_types, tgt_masks = torch.stack(padded_tgt_inputs), \
                                                            torch.stack(padded_tgt_positions), \
                                                            torch.stack(padded_tgt_token_types), \
                                                            torch.stack(padded_tgt_masks)

    inputs, positions, token_types, masks = torch.stack(padded_inputs), \
                                            torch.stack(padded_positions), \
                                            torch.stack(padded_token_types), \
                                            torch.stack(padded_masks)

    labels = labels.cuda(config.device)

    src_inputs = src_inputs.cuda(config.device)
    src_positions = src_positions.cuda(config.device)
    src_token_types = src_token_types.cuda(config.device)
    src_masks = src_masks.cuda(config.device)

    tgt_inputs = tgt_inputs.cuda(config.device)
    tgt_positions = tgt_positions.cuda(config.device)
    tgt_token_types = tgt_token_types.cuda(config.device)
    tgt_masks = tgt_masks.cuda(config.device)

    inputs = inputs.cuda(config.device)
    positions = positions.cuda(config.device)
    token_types = token_types.cuda(config.device)
    masks = masks.cuda(config.device)

    return [[src_inputs, src_positions, src_token_types, src_masks],
            [tgt_inputs, tgt_positions, tgt_token_types, tgt_masks],
            [inputs, positions, token_types, masks],
            labels]

def relationMask(ld, lr, li, lo):
    l_tot = ld + lr + li + lo
    mask = torch.zeros(l_tot, l_tot)
    # document to all
    mask[:ld, :] = torch.ones(ld, l_tot)
    # relation to relation, inputs, outputs
    mask[ld: ld + lr, ld:] = torch.ones(lr, l_tot - ld)
    # inputs to outputs
    mask[ld + lr:  ld + lr + li, ld + lr:] = torch.ones(li, l_tot - ld - lr)
    # outputs to everything behind it (including itself)
    mask[ld + lr + li:, ld + lr + li:] = torch.tril(torch.ones(lo, lo))
    return mask



def prepare_data_rel(docs, rels, inps, outps, tick, config):
    # [CLS] Document [SEP] Relation : Input Concept [SEP] Output Concept [SEP]
    ls = [len(doc) + len(rel) + len(inp) + len(outp) + 4 for doc, rel, inp, outp in zip(docs, rels, inps, outps)]
    l_max = max(ls)

    padded_inputs = []
    padded_positions = []
    padded_token_types = []

    padded_labels = []
    padded_masks = []

    for doc, rel, inp, outp, l in zip(docs, rels, inps, outps, ls):
        src = [config.CLS] + doc + [config.SEP] + rel + [config.SEP] + inp + [config.SEP]
        tgt = outp + [config.SEP]

        tgt_input, tgt_label = maskSeq(tgt, tick, config, True)

        input_i = torch.LongTensor(src + tgt_input)
        position_i = torch.cat(
            [torch.arange(len(doc) + 2),
             torch.arange(1, len(rel) + 2),
             torch.arange(1, len(inp) + 2),
             torch.arange(1, len(outp) + 2)]
        ).long()
        token_type_i = torch.LongTensor([0] * (len(doc) + 2) + [1] * (len(rel) + 1) + [2] * (len(inp) + 1) + [3] * (len(outp) + 1))
        label_i = torch.LongTensor([config.PAD] * len(src) + tgt_label)

        # source -> source, relation, inputs, outputs
        # relation -> relation, inputs
        # inputs -> input, outputs
        # outputs -> outputs
        mask_i = relationMask(len(doc) + 2, len(rel) + 1, len(inp) + 1, len(outp) + 1)

        # Padding
        padded_input_i = F.pad(input_i, (0, l_max - l), "constant", config.PAD)
        padded_position_i = F.pad(position_i, (0, l_max - l), "constant", 0)
        padded_token_type_i = F.pad(token_type_i, (0, l_max - l), "constant", 0)

        padded_label_i = F.pad(label_i, (0, l_max - l), "constant", config.PAD)
        padded_mask_i = F.pad(mask_i, (0, l_max - l, 0, l_max - l), "constant", 0.0)

        # Append to List

        padded_inputs.append(padded_input_i)
        padded_positions.append(padded_position_i)
        padded_token_types.append(padded_token_type_i)
        padded_labels.append(padded_label_i)
        padded_masks.append(padded_mask_i)

    inputs, positions, token_types, labels, masks = torch.stack(padded_inputs), \
                                                    torch.stack(padded_positions), torch.stack(padded_token_types), \
                                                    torch.stack(padded_labels), torch.stack(padded_masks)

    inputs = inputs.cuda(config.device)
    positions = positions.cuda(config.device)
    token_types = token_types.cuda(config.device)
    labels = labels.cuda(config.device)
    masks = masks.cuda(config.device)

    return [inputs, positions, token_types, labels, masks]

def prepare_data_new(srcs, tgts, tick, config):
    if config.padding_end >= 0:
        ls = [len(src) + 3 + config.gen_max_len for src in srcs]
    else:
        ls = [len(src) + len(tgt) + 3 for src, tgt in zip(srcs, tgts)]
    l_max = max(ls)

    padded_inputs = []
    padded_positions = []
    padded_token_types = []

    padded_labels = []
    padded_masks = []

    for src, tgt, l in zip(srcs, tgts, ls):
        src_ = [config.CLS] + src + [config.SEP]
        tgt_ = tgt + [config.SEP]

        if config.padding_end >= 0:
            while len(tgt_) <= config.gen_max_len:
                tgt_.append(config.padding_end)

        # Mask Both Sides
        new_src_input, new_src_label = maskSeq(src_, tick, config, False)
        new_tgt_input, new_tgt_label = maskSeq(tgt_, tick, config, True)

        # Cast to Torch Types

        input_i = torch.LongTensor(new_src_input + new_tgt_input)
        position_i = torch.cat([torch.arange(len(src_)), torch.arange(1, len(tgt_) + 1)]).long()
        token_type_i = torch.LongTensor([0] * (len(src_)) + [1] * len(tgt_))
        label_i = torch.LongTensor(new_src_label + new_tgt_label)

        mask_i = text2textMask(len(src_), len(tgt_))

        # Padding
        padded_input_i = F.pad(input_i, (0, l_max - l), "constant", config.PAD)
        padded_position_i = F.pad(position_i, (0, l_max - l), "constant", 0)
        padded_token_type_i = F.pad(token_type_i, (0, l_max - l), "constant", 0)

        padded_label_i = F.pad(label_i, (0, l_max - l), "constant", config.PAD)
        padded_mask_i = F.pad(mask_i, (0, l_max - l, 0, l_max - l), "constant", 0.0)

        # Append to list

        padded_inputs.append(padded_input_i)
        padded_positions.append(padded_position_i)
        padded_token_types.append(padded_token_type_i)

        padded_labels.append(padded_label_i)
        padded_masks.append(padded_mask_i)

    inputs, positions, token_types, labels, masks = torch.stack(padded_inputs), \
                                                    torch.stack(padded_positions), torch.stack(padded_token_types), \
                                                    torch.stack(padded_labels), torch.stack(padded_masks)

    #if config.parallel:
    inputs = inputs.cuda(config.device)
    positions = positions.cuda(config.device)
    token_types = token_types.cuda(config.device)
    labels = labels.cuda(config.device)
    masks = masks.cuda(config.device)

    return [inputs, positions, token_types, labels, masks]

def prepare_test_data(src, tgt, config):
    src = [config.CLS] + src + [config.SEP]
    tgt = tgt

    inputs = torch.LongTensor(src + tgt)
    positions = torch.cat([torch.arange(len(src)), torch.arange(1, len(tgt) + 1)]).long()
    token_types = torch.LongTensor([0] * len(src) + [1] * len(tgt))
    masks = text2textMask(len(src), len(tgt))
    output_mask = (inputs != config.MASK).float() * -10000.0

    inputs, positions, token_types, masks, output_mask = inputs.unsqueeze(0), positions.unsqueeze(0), token_types.unsqueeze(
        0), masks.unsqueeze(0), output_mask.unsqueeze(0)

    inputs = inputs.cuda(config.device)
    positions = positions.cuda(config.device)
    token_types = token_types.cuda(config.device)
    masks = masks.cuda(config.device)
    output_mask = output_mask.cuda(config.device)
    return [inputs, positions, token_types, masks, output_mask]

def prepare_test_data_new(src, tgt, config):
    src = [config.CLS] + src + [config.SEP]
    tgt = tgt + [config.SEP]

    inputs = torch.LongTensor(src + tgt)
    positions = torch.cat([torch.arange(len(src)), torch.arange(1, len(tgt) + 1)]).long()
    token_types = torch.LongTensor([0] * len(src) + [1] * len(tgt))
    masks = text2textMask(len(src), len(tgt))
    output_mask = (inputs != config.MASK).float() * -10000.0

    inputs, positions, token_types, masks, output_mask = inputs.unsqueeze(0), positions.unsqueeze(0), token_types.unsqueeze(
        0), masks.unsqueeze(0), output_mask.unsqueeze(0)

    inputs = inputs.cuda(config.device)
    positions = positions.cuda(config.device)
    token_types = token_types.cuda(config.device)
    masks = masks.cuda(config.device)
    output_mask = output_mask.cuda(config.device)
    return [inputs, positions, token_types, masks, output_mask]

def prepare_test_rel(src, tgt, config):
    doc, rel, inp, = src
    source = [config.CLS] + doc + [config.SEP] + rel + [config.SEP] + inp + [config.SEP]
    target = tgt + [config.MASK]

    inputs = torch.LongTensor(source + target)
    positions = torch.cat(
        [torch.arange(len(doc) + 2),
         torch.arange(1, len(rel) + 2),
         torch.arange(1, len(inp) + 2),
         torch.arange(1, len(tgt) + 2)]
    ).long()

    token_types = torch.LongTensor([0] * (len(doc) + 2) +
                                   [1] * (len(rel) + 1) +
                                   [2] * (len(inp) + 1) +
                                   [3] * (len(tgt) + 1))

    masks = relationMask(len(doc) + 2, len(rel) + 1, len(inp) + 1, len(tgt) + 1)
    inputs, positions, token_types, masks = inputs.unsqueeze(0), positions.unsqueeze(0), token_types.unsqueeze(0), masks.unsqueeze(0)

    inputs = inputs.cuda(config.device)
    positions = positions.cuda(config.device)
    token_types = token_types.cuda(config.device)
    masks = masks.cuda(config.device)

    return inputs, positions, token_types, masks



def nGram(seq, n):
    return list(zip(*[seq[i:] for i in range(n)]))


def do_tricks(preds, source, target, config):
    ban_ids = []
    if config.triGramTrick and len(target) > 2:
        current_triGrams = nGram(target, 3)
        for triGram in current_triGrams:
            if (target[-2] == triGram[0]) and (target[-1] == triGram[1]):
                ban_ids.append(triGram[2])

    ratio = 0.0
    bonus_ids = []
    if config.biGramTrick and len(target) > 0:
        bi_in = set(nGram(source, 2))
        bi_now = set(nGram(target, 2))
        available_biGrams = bi_in - bi_now

        for biGram in list(available_biGrams):
            if (target[-1] == biGram[0]):
                bonus_ids.append(biGram[1])

        ratio = config.gamma_value / (len(bi_in) + 1e-8)

    for idx in bonus_ids:
        # preds[idx] = min(0, preds[idx] + ratio)
        preds[idx] += ratio

    for idx in ban_ids:
        preds[idx] = -1e9

    return preds


def format_(x, y):
    fx = open(x, 'r')
    fy = open(y, 'w')
    for l in fx:
        line = l.lower().strip()
        print(line, file=fy)
    fx.close()
    fy.close()

def mapping_tokenize(s, t):
    st = 0
    ed = 0
    mapping = []
    mapping_idx = []
    for idx, token in enumerate(s):
        token_ = token.lower()
        prefix = "".join([piece.replace('##', '') for piece in t[st:ed + 1]])
        # print(prefix, type(prefix))
        while token_.startswith(prefix):
            ed += 1
            if ed >= len(t):
                break
            prefix = "".join([piece.replace('##', '') for piece in t[st:ed + 1]])
            # print(prefix, type(prefix))
        if (ed - st > 1) or (sum(1 for c in token if c.isupper()) > 1) or (idx > 0):
            mapping_idx.append([(st, ed), idx])
            mapping.append([cp(t[st:ed]), token])
        st = ed
    return mapping

def detokenize(text, mapping):
    if mapping is None:
        return text
    text = " " + text
    for one_mapping in mapping:
        keys = "".join([key.replace('##', '') if key.startswith('##') else ' ' + key for key in one_mapping[0]])
        value = ' ' + one_mapping[1]
        text = text.replace(keys, value)
    text = list(text[1:])
    if len(text) > 0:
        text[0] = text[0].upper()
        text = "".join(text)
    return text
