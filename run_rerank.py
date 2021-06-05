from data_process import myTokenizer
import argparse, math
from utility import loadFromPKL


def argLoader():
    parser = argparse.ArgumentParser()

    # Pre-training Model
    parser.add_argument('--pre_training_model', type=str, default='roberta-base')
    parser.add_argument('--padding', type=str, default="none")

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

    print(args)
    return args

def lengthReward(reward, length, bound):
    return reward * min(length, bound)

if __name__ == '__main__':
    N = 1951
    eps = 1e-6
    Threshold = 0.54
    reward = 2.0
    f_doc = open("./data/gigaword/test.Ndocument", "r")
    decoded = loadFromPKL("decoded.pkl")
    f_length = open("length.txt", "r")
    f_out = open("summary_rerank.txt", "w")
    config = argLoader()
    Tokenizer = myTokenizer(config)
    for i in range(N):
        Answers = decoded[i]
        nll = [Ans[0] for Ans in Answers]
        tokens = [Ans[1] for Ans in Answers]
        order = [Ans[2] for Ans in Answers]
        lengths = [len(token) for token in tokens]

        l_pred = eval(f_length.readline().strip())
        src = f_doc.readline().strip()

        M = len(tokens)
        score = [nllScore + lengthReward(reward, length, l_pred) for nllScore, length in zip(nll, lengths)]
        bestS = max(score)
        bestj = score.index(bestS)

        if lengths[bestj] < 3:
            text = src
        else:
            text = Tokenizer.decode(tokens[bestj])
        print(text)
        print(text, file=f_out)
    f_out.close()