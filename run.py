import gc
import time
import argparse
import torch
import random
import numpy as np

from mylog import mylog
from parallel import DataParallelModel, DataParallelCriterion
from model_bert import myBertForMaskedLM
from model_roberta import myRobertaForMaskedLM
from loss import KLDivLoss
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from utility import prepare_data_new, save_check_point, saveToPKL, loadFromJson, mapping_tokenize, detokenize
from data_process import myDataSet_pretrained as Dataset
from data_process import myTokenizer
from searcher import newSearcher

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
    parser.add_argument('--dataset', type=str, default='gigaword')
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
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    # newsroom settings warmup_steps 1000
    parser.add_argument('--batch_size', type=int, default=128)

    # Loss Parameters
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    # Corruption Parameters
    parser.add_argument('--corruption_schedule', type=str, default='linear')
    # can be constant, linear, triangle
    # for constant
    parser.add_argument('--corruption_source', type=float, default=0.1)
    parser.add_argument('--corruption_target', type=float, default=0.9)

    # for both linear and triangle
    parser.add_argument('--corruption_source_start', type=float, default=0.1)
    parser.add_argument('--corruption_source_end', type=float, default=0.0)
    parser.add_argument('--corruption_target_start', type=float, default=0.9)
    parser.add_argument('--corruption_target_end', type=float, default=0.6)

    # for both linear and triangle
    # newsroom setting 1454 * 20 = 29080
    parser.add_argument('--corruption_source_period', type=int, default=482460)
    parser.add_argument('--corruption_target_period', type=int, default=482460)

    parser.add_argument('--corruption_mask_rates', type=float, default=0.8)
    parser.add_argument('--corruption_random_rates', type=float, default=0.1)

    # Training Parameters
    parser.add_argument('--max_epoch', type=int, default=20)
    parser.add_argument('--checkPoint_Min', type=int, default=0)
    parser.add_argument('--checkPoint_Freq', type=int, default=1000)
    #newsroom settings 200

    parser.add_argument('--reduce_bound', type=int, default=100000000)
    parser.add_argument('--padding', type=str, default="none")
    parser.add_argument('--save_each_epoch', action="store_true")

    # Testting Parameters
    parser.add_argument('--model', type=str, default='./model/model_best_gen.pth.tar')
    parser.add_argument('--input', type=str, default='../../dataset/gigaword/test4.Ndocument')
    parser.add_argument('--standard', type=str, default='../../dataset/gigaword/test4.Nsummary')
    parser.add_argument('--search_method', type=str, default='lengthBeam')
    parser.add_argument('--rerank_method', type=str, default='smooth_bound_reward')

    parser.add_argument('--beam_size', type=int, default=20)
    parser.add_argument('--cands_limit', type=int, default=1000000)
    parser.add_argument('--answer_size', type=int, default=1)
    parser.add_argument('--gen_min_len', type=int, default=9)
    parser.add_argument('--gen_max_len', type=int, default=11)
    parser.add_argument('--gamma_value', type=float, default=14.0)
    parser.add_argument('--beta_value', type=float, default=0.5)
    parser.add_argument('--reward', type=float, default=0.25)
    parser.add_argument('--no_biGramTrick', action='store_true', help='Wheter do not biGramTrick')
    parser.add_argument('--no_triGramTrick', action='store_true', help='Wheter do not triGramTrick')

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

    if args.padding == "none":
        args.padding_end = -1
    elif args.padding == "pad":
        args.padding_end = args.PAD
    elif args.padding == "unk":
        args.padding_end = args.UNK
    elif args.padding == "sep":
        args.padding_end = args.SEP
    elif args.padding == "cls":
        args.padding_end = args.CLS

    if args.do_train:
        args.dataOptions = loadFromJson("settings/dataset/" + str(args.dataset) + ".json")
    elif args.do_test:
        args.biGramTrick = not args.no_biGramTrick
        args.triGramTrick = not args.no_triGramTrick

    print(args)
    return args

def train(config):
    # Model
    net = None
    if config.pre_training_model == "bert-base-uncased":
        net = myBertForMaskedLM.from_pretrained(config.pre_training_model)
    elif config.pre_training_model == "roberta-base":
        net = myRobertaForMaskedLM.from_pretrained(config.pre_training_model)

    lossFunc = KLDivLoss(config)
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
            srcs, tgts = batch_data
            inputs, positions, token_types, labels, masks = prepare_data_new(srcs, tgts, ticks, config)

            n_token = int((labels.data != config.PAD).data.sum())

            net.train()
            predicts = net(inputs, positions, token_types, masks)
            loss = lossFunc(predicts, labels, n_token).sum()

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
                nv_token = 0
                for bid, batch_data in enumerate(validSet):
                    srcs, tgts = batch_data
                    inputs, positions, token_types, labels, masks = prepare_data_new(srcs, tgts, -1, config)
                    n_token = int((labels.data != config.PAD).data.sum())
                    nv_token += n_token
                    with torch.no_grad():
                        net.eval()
                        predicts = net(inputs, positions, token_types, masks)
                        vloss += float(lossFunc(predicts, labels).sum())
                vloss /= nv_token
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

    net = None
    if config.pre_training_model == "bert-base-uncased":
        net = myBertForMaskedLM.from_pretrained(config.pre_training_model)
    elif config.pre_training_model == "roberta-base":
        net = myRobertaForMaskedLM.from_pretrained(config.pre_training_model)

    if torch.cuda.is_available():
        net = net.cuda(config.device)
        if config.parallel:
            net = DataParallelModel(net)

    net.load_state_dict(best_model["state_dict"])
    net.eval()

    mySearcher = newSearcher(net, config)

    f_in = open(config.input, 'r')
    output_files = {}
    order_files = {}
    for l in range(config.gen_min_len, config.gen_max_len + 1):
        output_files[l] = open("summary_" + str(l) + ".txt", "w")
        order_files[l] = open("order_" + str(l) + ".txt", "w")

    decoded = []

    for idx, line in enumerate(f_in):

        source_ = line.strip().split()
        source = Tokenizer.tokenize(line.strip())
        mapping = mapping_tokenize(source_, source)

        source = Tokenizer.encode(line.strip())

        print(idx)
        print(Tokenizer.decode(source))

        para = {}
        if config.search_method == "lengthBeam":
            para = {
                "minL": config.gen_min_len,
                "maxL": config.gen_max_len,
            }

        Answers = mySearcher.search(source, **para)
        decoded.append(Answers)

        for l in range(config.gen_min_len, config.gen_max_len + 1):
            for Ans in Answers:
                if len(Ans[1]) == l:
                    text = Tokenizer.decode(Ans[1], mapping)
                    tokens = Ans[1]
                    orders = Ans[2]
                    print(Ans)
                    print(text)
                    print(text, file=output_files[l])
                    print(tokens, file=order_files[l])
                    print(orders, file=order_files[l])

    saveToPKL("decoded.pkl", decoded)

    f_in.close()
    for f in output_files.values():
        f.close()

    for f in order_files.values():
        f.close()

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