from data_process import myTokenizer
import argparse, math


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
    parser.add_argument('--learning_rate', type=float, default=4e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_steps', type=int, default=10000)
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
    parser.add_argument('--corruption_source_period', type=int, default=241230)
    parser.add_argument('--corruption_target_period', type=int, default=24123)

    parser.add_argument('--corruption_mask_rates', type=float, default=0.8)
    parser.add_argument('--corruption_random_rates', type=float, default=0.1)

    # Training Parameters
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--checkPoint_Min', type=int, default=0)
    parser.add_argument('--checkPoint_Freq', type=int, default=1000)
    parser.add_argument('--reduce_bound', type=int, default=15000)
    parser.add_argument('--padding', type=str, default="none")
    parser.add_argument('--save_each_epoch', action="store_true")

    # Testting Parameters
    parser.add_argument('--model', type=str, default='./model/model_best.pth.tar')
    parser.add_argument('--input', type=str, default='../../dataset/gigaword/test4.Ndocument')
    parser.add_argument('--standard', type=str, default='../../dataset/gigaword/test4.Nsummary')
    parser.add_argument('--search_method', type=str, default='lengthBeam')
    parser.add_argument('--rerank_method', type=str, default='smooth_bound_reward')

    parser.add_argument('--beam_size', type=int, default=20)
    parser.add_argument('--cands_limit', type=int, default=1000000)
    parser.add_argument('--answer_size', type=int, default=1)
    parser.add_argument('--gen_min_len', type=int, default=30)
    parser.add_argument('--gen_max_len', type=int, default=30)
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


    print(args)
    return args

def neg_sigmoid(x):
    return 1.0 / (1 + math.exp(x))

'''
def lengthReward(reward, length, bound):
    reward_func = lambda x: reward * neg_sigmoid(x - bound)
    rewards = 0
    for i in range(length):
        rewards += reward_func(i)
    return rewards
'''

def lengthReward(reward, length, bound):
    return reward * min(length, bound)

if __name__ == '__main__':
    N = 1951
    eps = 1e-6
    Threshold = 0.54
    reward = 2.0
    f_doc = open("../../dataset/gigaword/test4.Ndocument", "r")
    f_in = open("all.txt", "r")
    f_length = open("length.txt", "r")
    f_out = open("trick.txt", "w")
    config = argLoader()
    Tokenizer = myTokenizer(config)
    total = 0
    for i in range(N):
        texts = eval(f_in.readline().strip())
        nll = eval(f_in.readline().strip())
        cands = eval(f_in.readline().strip())
        order = eval(f_in.readline().strip())
        rs = eval(f_in.readline().strip())
        #cands = [Tokenizer.encode(item) for item in cands]
        lengths = [len(cand) for cand in cands]
        l_pred = eval(f_length.readline().strip())
        src = f_doc.readline().strip()
        M = len(cands)
        score = [nllScore + lengthReward(reward, length, l_pred) for nllScore, length in zip(nll, lengths)]
        bestS = max(score)
        bestj = score.index(bestS)

        """
        Table = [0] * config.n_vocab
        for cand in cands:
            Vis = [False] * config.n_vocab
            for token in cand:
                if not Vis[token]:
                    Table[token] += 1
                    Vis[token] = True
        Tokens = []
        for idx, cnt in enumerate(Table):
            if cnt > 0:
                Tokens.append([idx, cnt / M])

        mapTokens = {}
        for item in Tokens:
            mapTokens[item[0]] = item[1]

        bestS = 0
        bestj = 0
        for j, cand in enumerate(cands):
            score = 0
            for token in cand:
                score += mapTokens[token]
            if score / len(cand) > bestS:
                bestS = score / len(cand)
                bestj = j
        """
        '''
        Tokens = sorted(Tokens, key=lambda x:-x[1])
        print(Tokens)
        SelectedTokens = [item for item in Tokens if item[1] >= Threshold]
        print(SelectedTokens)

        bestF = 0
        bestP = 0
        bestR = 0
        bestj = 0
        for j in range(M):
            cnt = 0
            for token in SelectedTokens:
                if token[0] in cands[j]:
                    cnt += 1
            P = cnt / (len(cands[j]) + eps)
            R = cnt / (len(SelectedTokens) + eps)
            F = 2 * P * R / (P + R + eps)
            #print(cnt, P, R, F)
            if F > bestF:
                bestF = F
                bestP = P
                bestR = R
                bestj = j

        bestRs = 0
        bestk = 0
        for k in range(M):
            if rs[k] > bestRs:
                bestRs = rs[k]
                bestk = k
        print(bestF, bestP, bestR, len(cands[bestj]), len(cands[bestk]))
        '''
        total += rs[bestj]
        print(bestj, bestS, total / (i+1))
        if len(cands[bestj]) < 3:
            text = src
        else:
            text = Tokenizer.decode(cands[bestj])
        print(text)
        print(text, file=f_out)

    f_in.close()
    f_out.close()