from sklearn.utils import shuffle
import pickle


def read_TREC():
    data = {}

    def read(mode):
        x, y = [], []

        with open("data/TREC/TREC_" + mode + ".txt", "r", encoding="utf-8") as f:
            for line in f:
                if line[-1] == "\n":
                    line = line[:-1]
                y.append(line.split()[0].split(":")[0])
                x.append(line.split()[1:])

        x, y = shuffle(x, y)

        if mode == "train":
            dev_idx = len(x) // 10
            data["dev_x"], data["dev_y"] = x[:dev_idx], y[:dev_idx]
            data["train_x"], data["train_y"] = x[dev_idx:], y[dev_idx:]
        else:
            data["test_x"], data["test_y"] = x, y

    read("train")
    read("test")

    return data


def read_SST1():
    data = {}

    def read(mode):
        x, y = [], []

        with open("data/SST1/stsa.fine." + mode, "r", encoding="utf-8") as f:
            for line in f:
                if line[-1] == "\n":
                    line = line[:-1]
                y.append(line.split()[0])
                x.append(line.split()[1:])

        x, y = shuffle(x, y)

        if mode == "train":

            with open("data/SST1/stsa.fine.phrases.train", "r", encoding="utf-8") as f:
                for line in f:
                    if line[-1] == "\n":
                        line = line[:-1]
                    y.append(line.split()[0])
                    x.append(line.split()[1:])

            x, y = shuffle(x, y)
            data["train_x"], data["train_y"] = x, y

        elif mode == "dev":
            data["dev_x"], data["dev_y"] = x, y
        else:
            data["test_x"], data["test_y"] = x, y

    read("train")
    read("dev")
    read("test")

    return data


def read_SST2():
    data = {}

    def read(mode):
        x, y = [], []

        with open("data/SST2/stsa.binary." + mode, "r", encoding="utf-8") as f:
            for line in f:
                if line[-1] == "\n":
                    line = line[:-1]
                y.append(line.split()[0])
                x.append(line.split()[1:])

        x, y = shuffle(x, y)

        if mode == "train":

            with open("data/SST2/stsa.binary.phrases.train", "r", encoding="utf-8") as f:
                for line in f:
                    if line[-1] == "\n":
                        line = line[:-1]
                    y.append(line.split()[0])
                    x.append(line.split()[1:])

            x, y = shuffle(x, y)
            data["train_x"], data["train_y"] = x, y

        elif mode == "dev":
            data["dev_x"], data["dev_y"] = x, y
        else:
            data["test_x"], data["test_y"] = x, y

    read("train")
    read("dev")
    read("test")

    return data


def read_MR():
    data = {}
    x, y = [], []

    with open("data/MR/rt-polarity.pos", "r", encoding="utf-8") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            x.append(line.split())
            y.append(1)

    with open("data/MR/rt-polarity.neg", "r", encoding="utf-8") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            x.append(line.split())
            y.append(0)

    x, y = shuffle(x, y)
    dev_idx = len(x) // 10 * 8
    test_idx = len(x) // 10 * 9

    data["train_x"], data["train_y"] = x[:dev_idx], y[:dev_idx]
    data["dev_x"], data["dev_y"] = x[dev_idx:test_idx], y[dev_idx:test_idx]
    data["test_x"], data["test_y"] = x[test_idx:], y[test_idx:]

    return data


def read_CR():
    data = {}
    x, y = [], []

    with open("data/CR/custrev.pos", "r", encoding="utf-8") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            x.append(line.split())
            y.append(1)

    with open("data/CR/custrev.neg", "r", encoding="utf-8") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            x.append(line.split())
            y.append(0)

    x, y = shuffle(x, y)
    dev_idx = len(x) // 10 * 8
    test_idx = len(x) // 10 * 9

    data["train_x"], data["train_y"] = x[:dev_idx], y[:dev_idx]
    data["dev_x"], data["dev_y"] = x[dev_idx:test_idx], y[dev_idx:test_idx]
    data["test_x"], data["test_y"] = x[test_idx:], y[test_idx:]

    return data


def read_MPQA():
    data = {}
    x, y = [], []

    with open("data/MPQA/mpqa.pos", "r", encoding="utf-8") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            x.append(line.split())
            y.append(1)

    with open("data/MPQA/mpqa.neg", "r", encoding="utf-8") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            x.append(line.split())
            y.append(0)

    x, y = shuffle(x, y)
    dev_idx = len(x) // 10 * 8
    test_idx = len(x) // 10 * 9

    data["train_x"], data["train_y"] = x[:dev_idx], y[:dev_idx]
    data["dev_x"], data["dev_y"] = x[dev_idx:test_idx], y[dev_idx:test_idx]
    data["test_x"], data["test_y"] = x[test_idx:], y[test_idx:]

    return data


def read_Subj():
    data = {}
    x, y = [], []

    with open("data/Subj/subj.objective", "r", encoding="utf-8") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            x.append(line.split())
            y.append(1)

    with open("data/Subj/subj.subjective", "r", encoding="utf-8") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            x.append(line.split())
            y.append(0)

    x, y = shuffle(x, y)
    dev_idx = len(x) // 10 * 8
    test_idx = len(x) // 10 * 9

    data["train_x"], data["train_y"] = x[:dev_idx], y[:dev_idx]
    data["dev_x"], data["dev_y"] = x[dev_idx:test_idx], y[dev_idx:test_idx]
    data["test_x"], data["test_y"] = x[test_idx:], y[test_idx:]

    return data


def save_model(model, params):
    path = "saved_models/{params['DATASET']}_{params['MODEL']}_{params['EPOCH']}.pkl".format(params=params)
    pickle.dump(model, open(path, "wb"))
    print("A model is saved successfully as {path}".format(path=path))


def load_model(params):
    path = "saved_models/{params['DATASET']}_{params['MODEL']}_{params['EPOCH']}.pkl".format(params=params)

    try:
        model =pickle.load(open(path, "rb"))
        print("Model in {path} loaded successfully".format(path=path))

        return model
    except:
        print("No available model such as {path}".format(path=path))
        exit()


