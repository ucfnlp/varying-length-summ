import torch.utils.data as Data
import random
from utility import Sentence2ListOfIndex, remove_digits, cp

class clsDataset(Data.Dataset):
    def __init__(self, name, batch_size, lenFunc, Tokenizer, options, log, mode='train'):
        self.name = name
        self.log = log
        self.batch_size = batch_size
        self.len_func = lenFunc
        self.Tokenizer = Tokenizer
        self.mode = mode

        self.path = options['Parts'][name]['path']
        self.sorted = options['Parts'][name]['sorted']
        self.shuffled = options['Parts'][name]['shuffled']

        self.source_len = options['max_input_len']
        self.target_len = options['max_output_len']
        self.match = options['match']

        self.n_data = 0
        self.Data = []
        self.n_batch = 0
        self.Batch = []
        self.Batch_idx = []

        self.log.log('Building dataset %s from original text documents' % (self.name))
        self.n_data, self.Data = self.load()
        self.log.log('Finish Loading dataset %s' % (self.name))

        self.afterLoad()

    def sortByLength(self):
        self.log.log('Start sorting by length')
        data = self.Data
        number = self.n_data

        lengths = [(self.len_func(data[Index]), Index) for Index in range(number)]
        sorted_lengths = sorted(lengths)
        sorted_Index = [d[1] for d in sorted_lengths]

        data_new = [data[sorted_Index[Index]] for Index in range(number)]

        self.Data = data_new
        self.log.log('Finish sorting by length')

    def shuffle(self):
        self.log.log('Start Shuffling')

        data = self.Data
        number = self.n_data

        shuffle_Index = list(range(number))
        random.shuffle(shuffle_Index)

        data_new = [data[shuffle_Index[Index]] for Index in range(number)]

        self.Data = data_new
        self.log.log('Finish Shuffling')

    def genBatches(self):
        batch_size = self.batch_size
        data = self.Data
        number = self.n_data
        n_dim = len(data[0])

        number_batch = number // batch_size
        batches = []

        for bid in range(number_batch):
            batch_i = []
            for j in range(n_dim):
                data_j = [item[j] for item in data[bid * batch_size: (bid + 1) * batch_size]]
                batch_i.append(data_j)
            batches.append(batch_i)

        if number_batch * batch_size < number:
            batch_i = []
            for j in range(n_dim):
                data_j = [item[j] for item in data[number_batch * batch_size:]]
                batch_i.append(data_j)
            batches.append(batch_i)
            number_batch += 1

        self.n_batch = number_batch
        self.Batch = batches
        self.Batch_idx = list(range(self.n_batch))

    def load(self):
        srcFile = open(self.path + '_input.txt', 'r', encoding='utf-8')
        refFile = open(self.path + '_output.txt', 'r', encoding='utf-8')
        labFile = open(self.path + '_label.txt', 'r', encoding='utf-8')
        data = []

        Index = 0
        while True:
            Index += 1
            srcLine = srcFile.readline()
            if not srcLine:
                break
            refLine = refFile.readline()
            if not refLine:
                break
            labLine = labFile.readline()
            if not labLine:
                break
            srcLine = srcLine.strip()
            refLine = refLine.strip()
            labLine = labLine.strip()
            src_tokens = srcLine.split()
            ref_tokens = refLine.split()
            label = int(labLine)

            if (len(src_tokens) < 1) or (len(ref_tokens) < 1):
                continue

            if self.match and ('Train' in self.name):
                match = len(set(src_tokens) & set(ref_tokens))
                if match < 3:
                    continue

            if len(src_tokens) > self.source_len:
                src_tokens = src_tokens[:self.source_len]
                srcLine = " ".join(src_tokens)

            if len(ref_tokens) > self.target_len:
                ref_tokens = ref_tokens[:self.target_len]
                refLine = " ".join(ref_tokens)

            document = self.Tokenizer.encode(srcLine)
            summary = self.Tokenizer.encode(refLine)

            if len(document) > 90:
                document = document[:90]

            if len(summary) > 30:
                summary = summary[:30]

            data.append([document, summary, label])

        return len(data), data

    def afterLoad(self):
        if self.sorted:
            self.sortByLength()
        if self.shuffled:
            self.shuffle()

        # Generate Batches
        self.log.log('Generating Batches')
        self.genBatches()

    def batchShuffle(self):
        random.shuffle(self.Batch_idx)

    def __len__(self):
        if self.mode == 'train' or self.mode == 'valid':
            return self.n_batch
        return self.n_data

    def __getitem__(self, index):
        source, target, label = self.Batch[self.Batch_idx[index]]
        return [source, target, label]