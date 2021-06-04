import torch.utils.data as Data
import random, json

class relDataset(Data.Dataset):
    def __init__(self, name, batch_size, lenFunc, Tokenizer, options, log, mode="train"):
        self.name = name
        self.log = log
        self.batch_size = batch_size
        self.len_func = lenFunc
        self.Tokenizer = Tokenizer
        self.mode = mode

        self.path = options['Parts'][name]['path']
        self.sorted = options['Parts'][name]['sorted']
        self.shuffled = options['Parts'][name]['shuffled']

        self.n_data = 0
        self.Data = []
        self.n_batch = 0
        self.Batch = []
        self.Batch_idx = []

        self.log.log("Building dataset %s from original text." % (self.name))
        self.n_data, self.Data = self.load()
        self.log.log("Finish Loading dataset %s" % (self.name))

        self.afterLoad()

    def sortByLength(self):
        self.log.log('Start sorting by length.')
        data = self.Data
        number = self.n_data

        lengths = [(self.len_func(data[Index]), Index) for Index in range(number)]
        sorted_lengths = sorted(lengths)
        sorted_Index = [d[1] for d in sorted_lengths]

        data_new = [data[sorted_Index[Index]] for Index in range(number)]

        self.Data = data_new
        self.log.log('Finish sorting by length.')

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
        dataFile = open(self.path + '.json', 'r')
        data = []
        for index, line in enumerate(dataFile):
            data_i = json.loads(line)
            document = self.Tokenizer.encode(data_i["doc"])
            summary = self.Tokenizer.encode(data_i["ref"])
            relation = self.Tokenizer.encode(data_i["rel"])
            inputs = self.Tokenizer.encode(data_i["concept_1"])
            outputs = self.Tokenizer.encode(data_i["concept_2"])

            if len(document) + len(relation) + len(inputs) + len(outputs) >= 100:
                continue

            data.append([document, relation, inputs, outputs])
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
        document, relation, inputs, outputs = self.Batch[self.Batch_idx[index]]
        return [document, relation, inputs, outputs]