from TextCNN import CNN
import utils

from torch.autograd import  Variable
import torch
import torch.optim as optim
import torch.nn as nn

from sklearn.utils import shuffle
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import argparse
import copy


def train(data, params):
    if params["MODEL"] != "rand":
        # 加载 wor2vec
        print("loading word2vec...")
        word_vectors = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

        # 创建本文档的 word2vec 矩阵
        wv_matrix = []
        for i in range(len(data["vocab"])):
            word = data["idx_to_word"][i]
            # 判断 word 是不是存在于 word2vec 存在就直接使用， 不存在就随机初始化代替
            if word in word_vectors.vocab:
                wv_matrix.append(word_vectors.word_vec(word))
            else:
                wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))

        # one for UNK and one for zero padding
        wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
        wv_matrix.append(np.zeros(300).astype("float32"))
        wv_matrix = np.array(wv_matrix)
        params["WV_MATRIX"] = wv_matrix

    # 实例化模型 优化器 损失函数
    model = CNN(**params).cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters()) # 得到具备 requires_grad 的参数
    optimizer = optim.Adam(parameters, params["LEARNING_RATE"])
    criterion = nn.CrossEntropyLoss()

    pre_dev_acc = 0
    max_dev_acc = 0
    max_test_acc = 0

    for e in range(params["EPOCH"]):
        data["train_x"], data["train_y"] = shuffle(data["train_x"], data["train_y"])

        for i in range(0, len(data["train_x"]), params["BATCH_SIZE"]):
            batch_range = min(params["BATCH_SIZE"], len(data["train_x"]) - i)
            # 得到句子中所有单词在 vocabulary 的 idx , 以此去索引该词在本文档 word2vec 矩阵中的词向量( model 中实现)
            batch_x = [[data["word_to_idx"][w] for w in sent] +
                       [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
                       for sent in data["train_x"][i:i + batch_range]]
            batch_y = [data["classes"].index(c) for c in data["train_y"][i:i + batch_range]]

            batch_x = Variable(torch.LongTensor(batch_x)).cuda()
            batch_y = Variable(torch.LongTensor(batch_y)).cuda()

            optimizer.zero_grad()
            model.train()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm(parameters, max_norm=params["NORM_LIMIT"])
            optimizer.step()

        dev_acc = test(data, model, params, mode="dev")
        test_acc = test(data, model, params)
        print("epoch:", e + 1, "/ dev_acc:", dev_acc, "/ test_acc:", test_acc)

        if params["EARLY_STOPPING"] and dev_acc <= pre_dev_acc:
            print("early stopping by dev_acc!")
            break
        else:
            pre_dev_acc = dev_acc

        if dev_acc > max_dev_acc:
            max_dev_acc = dev_acc
            max_test_acc = test_acc
            best_model = copy.deepcopy(model)

    print("max dev acc:", max_dev_acc, "test acc:", max_test_acc)
    return best_model


def test(data, model, params, mode="test"):
    model.eval()

    if mode == "dev":
        x, y = data["dev_x"], data["dev_y"]
    elif mode == "test":
        x, y = data["test_x"], data["test_y"]

    x = [[data["word_to_idx"][w] if w in data["vocab"] else params["VOCAB_SIZE"] for w in sent] +
         [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
         for sent in x]

    x = Variable(torch.LongTensor(x)).cuda()
    y = [data["classes"].index(c) for c in y]

    pred = np.argmax(model(x).cpu().data.numpy(), axis=1)
    acc = sum([1 if p == y else 0 for p, y in zip(pred, y)]) / len(pred)

    return acc


def main():
    parser = argparse.ArgumentParser(description="-----Text Classification by CNN-----")
    parser.add_argument("--mode", default="train", help="train: train (with test) a model / test: test/dev")
    parser.add_argument("--model", default="rand", help="available models:rand, static, non-static, multichannel")
    parser.add_argument("--dataset", default="TREC", help="available datasets: MR, TREC, Subj, SST1, SST2, MPQA")
    parser.add_argument("--save_model", default=False, action="store_true")
    parser.add_argument("--early_stopping", default=False, action="store_true")
    parser.add_argument("--epoch", default=3, type=int)
    parser.add_argument("--learning_rate", default=1.0, type=float)

    options = parser.parse_args()
    # 读取指定的 data set
    data = getattr(utils, "read_{options.dataset}".format(options=options))()
    # 创建包含 train test dev 中所有单词的 vocabulary
    data["vocab"] = sorted(list(set([w for sent in data["train_x"] + data["dev_x"] + data["test_x"] for w in sent])))
    # 检查有多少待分类的类别
    data["classes"] = sorted(list(set(data["train_y"])))
    data["word_to_idx"] = {w: i for i, w in enumerate(data["vocab"])}
    data["idx_to_word"] = {i: w for i, w in enumerate(data["vocab"])}

    params = {
        "MODEL": options.model,
        "DATASET": options.dataset,
        "SAVE_MODEL": options.save_model,
        "EARLY_STOPPING": options.early_stopping,
        "EPOCH": options.epoch,
        "LEARNING_RATE": options.learning_rate,
        "MAX_SENT_LEN": max([len(sent) for sent in data["train_x"] + data["dev_x"] + data["test_x"]]),
        "BATCH_SIZE": 50,
        "WORD_DIM": 300,
        "VOCAB_SIZE": len(data["vocab"]),
        "CLASS_SIZE": len(data["classes"]),
        "FILTERS": [3, 4, 5],
        "FILTER_NUM": [100, 100, 100],
        "DROPOUT_PROB": 0.5,
        "NORM_LIMIT": 3,
    }

    print("=" * 20 + "INFORMATION" + "=" * 20)
    print("MODEL:", params["MODEL"])
    print("DATASET:", params["DATASET"])
    print("VOCAB_SIZE:", params["VOCAB_SIZE"])
    print("EPOCH:", params["EPOCH"])
    print("LEARNING_RATE:", params["LEARNING_RATE"])
    print("EARLY_STOPPING:", params["EARLY_STOPPING"])
    print("SAVE_MODEL:", params["SAVE_MODEL"])
    print("=" * 20 + "INFORMATION" + "=" * 20)

    if options.mode == "train":
        print("=" * 20 + "TRAINING STARTED" + "=" * 20)
        model = train(data, params)
        if params["SAVE_MODEL"]:
            utils.save_model(model, params)
        print("=" * 20 + "TRAINING FINISHED" + "=" * 20)
    else:
        model = utils.load_model(params).cuda()

        test_acc = test(data, model, params)
        print("test acc:", test_acc)


if __name__ == "__main__":
        main()
