import gc
import time
import argparse
import torch
import random
import numpy as np

from mylog import mylog
from parallel import DataParallelModel, DataParallelCriterion
from model_roberta import myRobertaForMaskedLM, RobertaForSequenceClassification
from loss import KLDivLoss, BCELoss
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from utility import prepare_data_cls, save_check_point, loadFromJson, mapping_tokenize, detokenize
from data_process import clsDataset as Dataset
from data_process import myTokenizer

# Setup Random Seeds

seed = 19940609
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Setup LOG File

LOG = mylog(reset=True)


def argLoader():
    parser = argparse.ArgumentParser()

    # Actions
    parser.add_argument('--do_train', action='store_true', help="Whether to run training")
    parser.add_argument('--do_test', action='store_true', help="Whether to run test")

    # Options Setting
    parser.add_argument('--dataset', type=str, default='gigaword_cls')
    parser.add_argument('--part', type=str, default='train')

    # Model Saving Setting
    parser.add_argument('--save_path', type=str, default='./model')

    # Pre-training Model
    parser.add_argument('--pre_training_model', type=str, default='roberta-base')

    # Device Parameters
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataLoader_workers', type=int, default=1)

    # Optimization Parameters
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_steps', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=128)

    # Loss Parameters
    parser.add_argument('--num_labels', type=int, default=1)

    # Training Parameters
    parser.add_argument('--max_epoch', type=int, default=4)
    parser.add_argument('--checkPoint_Min', type=int, default=0)
    parser.add_argument('--checkPoint_Freq', type=int, default=1000)
    parser.add_argument('--reduce_bound', type=int, default=100000000)
    parser.add_argument('--save_each_epoch', action="store_true")

    # Testting Parameters
    parser.add_argument('--model', type=str, default='./model/model_best_cls.pth.tar')
    parser.add_argument('--input', type=str, default='../../dataset/gigaword/test4.Ndocument')
    parser.add_argument('--standard', type=str, default='../../dataset/gigaword/test4.Nsummary')

    args = parser.parse_args()

    if args.pre_training_model == "bert-base-uncased":
        args.PAD = 0
        args.UNK = 100
        args.CLS = 101
        args.SEP = 102
        args.MASK = 103
        args.n_vocab = 30522
    elif args.pre_training_model == "roberta-base":
        args.PAD = 1
        args.UNK = 3
        args.CLS = 0
        args.SEP = 2
        args.MASK = 50264
        args.n_vocab = 50265

    if args.do_train or args.do_test:
        args.dataOptions = loadFromJson("settings/dataset/" + str(args.dataset) + ".json")

    print(args)
    return args

def train(config):
    # Model
    net = RobertaForSequenceClassification.from_pretrained(config.pre_training_model)
    lossFunc = BCELoss()

    if torch.cuda.is_available():
        net = net.cuda(config.device)
        lossFunc = lossFunc.cuda(config.device)

        if config.parallel:
            net = DataParallelModel(net)
            lossFunc = DataParallelCriterion(lossFunc)

    # Data options
    Tokenizer = myTokenizer(config)
    trainSet = Dataset(config.part, config.batch_size, lambda x: len(x[0]) + len(x[1]), Tokenizer, config.dataOptions, LOG, 'train')
    validSet = Dataset('valid', config.batch_size, lambda x: len(x[0]) + len(x[1]), Tokenizer, config.dataOptions, LOG, 'valid')

    # Learning Parameters
    num_batches_per_epoch = len(trainSet)
    learning_rate = config.learning_rate

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {"params": [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=config.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=num_batches_per_epoch * config.max_epoch
    )
    optimizer.zero_grad()

    ticks = 0
    Q = []
    best_vloss = 1e99
    counter = 0

    LOG.log("There are %d batches per epoch" % (len(trainSet)))

    for epoch_idx in range(config.max_epoch):
        trainSet.batchShuffle()
        LOG.log("Batch Shuffled")
        for batch_idx, batch_data in enumerate(trainSet):
            # release memory
            if (ticks + 1) % 1000 == 0:
                gc.collect()

            start_time = time.time()
            ticks += 1
            srcs, tgts, labels = batch_data
            src_inputs, tgt_inputs, all_inputs, labels = prepare_data_cls(srcs, tgts, labels, config)

            net.train()
            predicts = net(src_inputs, tgt_inputs, all_inputs)
            #print(predicts.size())
            #print(labels.size())
            batch_size = len(labels)
            loss = lossFunc(predicts, labels.view(-1, 1)).sum() / batch_size

            Q.append(float(loss))
            if len(Q) > 200:
                Q.pop(0)
            loss_avg = sum(Q) / len(Q)

            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            LOG.log('Epoch %2d, Batch %6d, Loss %9.6f, Average Loss %9.6f, Time %9.6f' %
                    (epoch_idx + 1, batch_idx + 1, loss, loss_avg, time.time() - start_time))

            loss = None

            # check points
            if (ticks >= config.checkPoint_Min) and (ticks % config.checkPoint_Freq == 0):
                gc.collect()
                vloss = 0
                total_size = 0
                for bid, batch_data in enumerate(validSet):
                    srcs, tgts, labels = batch_data
                    src_inputs, tgt_inputs, all_inputs, labels = prepare_data_cls(srcs, tgts, labels, config)
                    batch_size = len(labels)
                    total_size += batch_size
                    with torch.no_grad():
                        net.eval()
                        predicts = net(src_inputs, tgt_inputs, all_inputs)
                        vloss += float(lossFunc(predicts, labels.view(-1, 1)).sum())
                vloss /= total_size
                is_best = vloss < best_vloss
                best_vloss = min(vloss, best_vloss)
                LOG.log('CheckPoint: Validation Loss %11.8f, Best Loss %11.8f' % (vloss, best_vloss))
                vloss = None

                if is_best:
                    LOG.log('Best Model Updated')
                    save_check_point({
                        'epoch': epoch_idx + 1,
                        'batch': batch_idx + 1,
                        'config': config,
                        'state_dict': net.state_dict(),
                        'best_vloss': best_vloss},
                        is_best,
                        path=config.save_path,
                        fileName='latest.pth.tar'
                    )
                    counter = 0
                else:
                    counter += config.checkPoint_Freq
                    if counter >= config.reduce_bound:
                        counter = 0
                        for idx, base_lr in enumerate(scheduler.base_lrs):
                            scheduler.base_lrs[idx] = base_lr * 0.55
                            LOG.log('Reduce Base Learning Rate from %11.8f to %11.8f' % (base_lr, base_lr * 0.55))
                    LOG.log('Current Counter = %d' % (counter))

        if config.save_each_epoch:
            LOG.log('Saving Model after %d-th Epoch.' % (epoch_idx + 1))
            save_check_point({
                'epoch': epoch_idx + 1,
                'batch': batch_idx + 1,
                'config': config,
                'state_dict': net.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),
                'best_vloss': 1e99},
                False,
                path=config.save_path,
                fileName='checkpoint_Epoch' + str(epoch_idx + 1) + '.pth.tar'
            )

        LOG.log('Epoch Finished.')
        gc.collect()

def test(config):
    best_model = torch.load(config.model)
    Tokenizer = myTokenizer(config)

    net = RobertaForSequenceClassification.from_pretrained(config.pre_training_model)

    if torch.cuda.is_available():
        net = net.cuda(config.device)
        if config.parallel:
            net = DataParallelModel(net)

    net.load_state_dict(best_model["state_dict"])
    net.eval()

    testSet = Dataset('test500', 1, lambda x: len(x[0]) + len(x[1]), Tokenizer, config.dataOptions, LOG, 'test')

    f_predict = open("predict.txt", "w")
    for idx, batch_data in enumerate(testSet):
        srcs, tgts, labels = batch_data
        src_inputs, tgt_inputs, inputs, labels = prepare_data_cls(srcs, tgts, labels, config)
        with torch.no_grad():
            predicts = net(src_inputs, tgt_inputs, inputs)[0]
            predict = int(float(predicts.view(-1).data) >= 0.5)
            label = int(labels.view(-1).data)
            print(predict, file=f_predict)

    f_predict.close()

def main():
    args = argLoader()
    print("Totally", torch.cuda.device_count(), "GPUs are available.")
    if args.parallel:
        print("Using data parallel.")
        for device in range(torch.cuda.device_count()):
            print("Using #", device, "named", torch.cuda.get_device_name(device), "with", (torch.cuda.get_device_properties(device).total_memory-torch.cuda.memory_allocated(device)) // 1000 // 1000 / 1000, "GB Memory available.")
    else:
        torch.cuda.set_device(args.device)
        print("Using #", args.device , "named", torch.cuda.get_device_name(args.device), (torch.cuda.get_device_properties(args.device).total_memory-torch.cuda.memory_allocated(args.device)) // 1000 // 1000 / 1000, "GB Memory available.")

    if args.do_train:
        train(args)
    elif args.do_test:
        test(args)

if __name__ == '__main__':
    main()