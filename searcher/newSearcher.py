import torch
from copy import deepcopy as cp
from .splayTree import Splay
from utility import prepare_test_data_new

class newSearcher:
    def __init__(self, net, config):
        self.net = net
        self.config = config
        self.search_method = eval("self."+config.search_method)

    def Greedy(self, source):
        Answers = []

        minLen = self.config.gen_min_len
        maxLen = self.config.gen_max_len + 1
        for l in range(minLen, maxLen):
            target = [cp(self.config.MASK)] * l
            order = []
            for i in range(l):
                inputs, positions, token_types, masks, output_mask = prepare_test_data_new(source, target, self.config)
                with torch.no_grad():
                    predicts = self.net(inputs, positions, token_types, masks)
                    if type(predicts) == list:
                        predicts = predicts[0]
                    predicts = predicts[0][len(source) + 2:]
                    output_mask = output_mask[0][len(source) + 2:]
                    predicts_ = torch.max(predicts, dim=-1)[0] + output_mask
                    selected_pos = int(torch.argmax(predicts_))
                    selected_token = int(torch.argmax(predicts[selected_pos]))
                    target[selected_pos] = selected_token
                    order.append(selected_pos)
            inputs, positions, token_types, masks, output_mask = prepare_test_data_new(source, target, self.config)
            with torch.no_grad():
                predicts = self.net(inputs, positions, token_types, masks)
                if type(predicts) == list:
                    predicts = predicts[0]
                predicts = predicts[0][len(source) + 2:]
            score = 0
            for i in range(l):
                score += float(predicts[i][target[i]])
            Answers.append([score, cp(target), cp(order)])
        return Answers

    def lengthGreedy(self, source):
        Answers = []
        for l in range(self.config.gen_max_len, 0, -1):
            if (l == self.config.gen_max_len) or ((len(Answers) > 0) and (len(Answers[0][1]) > 1) and (Answers[0][1][-2] == self.config.SEP)):
                Answers = []
                target = [cp(self.config.MASK)] * l
                order = []
                for i in range(l):
                    inputs, positions, token_types, masks, output_mask = prepare_test_data_new(source, target, self.config)
                    with torch.no_grad():
                        predicts = self.net(inputs, positions, token_types, masks)
                        if type(predicts) == list:
                            predicts = predicts[0]
                        predicts = predicts[0][len(source) + 2:]
                        output_mask = output_mask[0][len(source) + 2:]
                        predicts_ = torch.max(predicts, dim=-1)[0] + output_mask
                        selected_pos = int(torch.argmax(predicts_))
                        selected_token = int(torch.argmax(predicts[selected_pos]))
                        target[selected_pos] = selected_token
                        order.append(selected_pos)
                inputs, positions, token_types, masks, output_mask = prepare_test_data_new(source, target, self.config)
                with torch.no_grad():
                    predicts = self.net(inputs, positions, token_types, masks)
                    if type(predicts) == list:
                        predicts = predicts[0]
                    predicts = predicts[0][len(source) + 2:]
                score = 0
                for i in range(l):
                    score += float(predicts[i][target[i]])
                Answers.append([score, cp(target), cp(order)])
            else:
                return Answers
        return Answers

    def Beam(self, source, l=31):
        Answers = []
        target = [cp(self.config.MASK)] * l
        startState = [0.0, target, []]
        Cands = [[startState]]
        for i in range(l):
            Cands_i = Cands[i]
            Cands_new = []
            for score, cand, order in Cands_i:
                inputs, positions, token_types, masks, output_mask = prepare_test_data_new(source, cand, self.config)
                with torch.no_grad():
                    predicts = self.net(inputs, positions, token_types, masks)
                    if type(predicts) == list:
                        predicts = predicts[0]
                    predicts = predicts[0][len(source) + 2:]
                    output_mask = output_mask[0][len(source) + 2:].unsqueeze(1)
                    predicts += output_mask
                    topk = torch.topk(predicts.view(-1), self.config.beam_size, sorted=True)
                    topk_value = topk[0]
                    topk_pos = topk[1] // self.config.n_vocab
                    topk_token = topk[1] - self.config.n_vocab * topk_pos

                    for posi, token, value in zip(topk_pos.tolist(), topk_token.tolist(), topk_value.tolist()):
                        new_score = score + float(value)
                        target = cp(cand)
                        target[int(posi)] = int(token)
                        flag = -1
                        for idx, item in enumerate(Cands_new):
                            if item[1] == target:
                                flag = idx
                                break
                        if flag < 0:
                            Cands_new.append([new_score, target, cp(order) + [int(posi)]])
                        elif Cands_new[idx][0] < new_score:
                            Cands_new[idx][0] = new_score
                            Cands_new[idx][2] = cp(order) + [int(posi)]
            Cands_new = sorted(Cands_new, key= lambda x: -x[0])
            if len(Cands_new) > self.config.beam_size:
                Cands_new = Cands_new[:self.config.beam_size]
            Cands.append(Cands_new)
        Answers += Cands[-1]
        return Answers

    def lengthBeam(self, source, minL=1, maxL=30):
        Answers = []
        minL = min(len(source), minL)
        maxL = min(len(source), maxL) + 1
        for l in range(minL, maxL):
            Answers_ = self.Beam(source, l)
            Answers_ = sorted(Answers_, key=lambda x: -x[0])
            Answers.append(Answers_[0])
        return Answers

    def greedyBeam(self, source):
        Greedy = self.Greedy(source)[0][1]
        L_pred = min(Greedy.index(self.config.SEP) + 3, self.config.gen_max_len)
        print("length prediction", L_pred)
        Answers = []
        target = [cp(self.config.MASK)] * self.config.gen_max_len
        for l in range(L_pred, self.config.gen_max_len):
            target[l] = cp(self.config.SEP)
        startState = [0.0, target, []]
        Cands = [[startState]]
        for i in range(L_pred):
            beam_size = max(self.config.beam_size, (L_pred - i))
            Cands_i = Cands[i]
            Cands_new = []
            for score, cand, order in Cands_i:
                inputs, positions, token_types, masks, output_mask = prepare_test_data_new(source, cand, self.config)
                with torch.no_grad():
                    predicts = self.net(inputs, positions, token_types, masks)
                    if type(predicts) == list:
                        predicts = predicts[0]
                    predicts = predicts[0][len(source) + 2:]
                    output_mask = output_mask[0][len(source) + 2:].unsqueeze(1)
                    predicts += output_mask
                    topk = torch.topk(predicts.view(-1), beam_size, sorted=True)
                    topk_value = topk[0]
                    topk_pos = topk[1] // self.config.n_vocab
                    topk_token = topk[1] - self.config.n_vocab * topk_pos

                    for posi, token, value in zip(topk_pos.tolist(), topk_token.tolist(), topk_value.tolist()):
                        new_score = score + float(value)
                        target = cp(cand)
                        target[int(posi)] = int(token)
                        flag = -1
                        for idx, item in enumerate(Cands_new):
                            if item[1] == target:
                                flag = idx
                                break
                        if flag < 0:
                            Cands_new.append([new_score, target, cp(order) + [int(posi)]])
                        elif Cands_new[idx][0] < new_score:
                            Cands_new[idx][0] = new_score
                            Cands_new[idx][2] = cp(order) + [int(posi)]
            Cands_new = sorted(Cands_new, key=lambda x: -x[0])
            if len(Cands_new) > beam_size:
                Cands_new = Cands_new[:beam_size]
            Cands.append(Cands_new)
        Answers += Cands[-1]
        return Answers

    def BestFirst(self, source):
        Answers = Splay(self.config.answer_size)
        Cands = Splay(self.config.cands_limit)

        target = [cp(self.config.MASK)] * self.config.gen_max_len
        startState = [0.0, target, []]
        Cands.push(0.0, startState)
        while Cands.isNotEmpty() and (Answers.size() < self.config.answer_size):
            head = Cands.pop()
            score, target, order = head.value
            #print(score, Cands.size())

            if self.config.MASK not in target:
                Answers.push(score, [score, target, order])
                continue

            inputs, positions, token_types, masks, output_mask = prepare_test_data_new(source, target, self.config)
            with torch.no_grad():
                predicts = self.net(inputs, positions, token_types, masks)
                if type(predicts) == list:
                    predicts = predicts[0]
                predicts = predicts[0][len(source) + 2:]
                output_mask = output_mask[0][len(source) + 2:].unsqueeze(1)
                predicts += output_mask
                topk = torch.topk(predicts.view(-1), self.config.beam_size, sorted=True)
                topk_value = topk[0]
                topk_pos = topk[1] // self.config.n_vocab
                topk_token = topk[1] - self.config.n_vocab * topk_pos
                for posi, token, value in zip(topk_pos.tolist(), topk_token.tolist(), topk_value.tolist()):
                    new_score = score - float(value)
                    new_target = cp(target)
                    new_target[int(posi)] = int(token)
                    Cands.push(new_score, [new_score, new_target, cp(order) + [int(posi)]])

        if Answers.size() < 1:
            print('No Answers')
            Answers.push(0.0, [[0.0], source])
        Answer_List = []
        while Answers.isNotEmpty():
            Answer_List.append(Answers.pop().value)
        return Answer_List


    def search(self, *args, **kwargs):
        return self.search_method(*args, **kwargs)