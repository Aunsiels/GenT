"""
This script evaluates the MRR, Precision@K and Recall@K for automatic triple alignment. It takes two arguments:
  * The generation file
  * The test file
"""
from sys import argv

GENERATIONS_FILENAME = argv[1]
TEST_SET_FILENAME = argv[2]


goldstandard = dict()
with open(TEST_SET_FILENAME) as f:
    for line in f:
        line = line.strip().strip("<|endoftext|>").split("[SEP]")
        if len(line) != 2:
            continue
        if line[0] not in goldstandard:
            goldstandard[line[0]] = []
        goldstandard[line[0]].append(line[1])


res = []
with open(GENERATIONS_FILENAME) as f:
    for line in f:
        line = line.strip().split("\t")
        original = line[0].replace(",", "\t")
        if original not in goldstandard:
            continue
        truth = goldstandard[original]
        temp_res = []
        for i, pred in enumerate(line[1:]):
            pred_tab = pred.replace(",", "\t")
            if pred_tab in truth:
                temp_res.append(i)
        res.append((temp_res, len(truth)))


mrrs = []
for pred, _ in res:
    temp = 0
    for rank in pred:
        temp = 1.0 / (rank + 1.0)
        break
    mrrs.append(temp)

print("MRR:", sum(mrrs) / len(mrrs))


for at in range(10):
    recall = []
    precision = []
    for pred, size in res:
        p_temp = 0
        for rank in pred:
            if rank <= at:
                p_temp += 1
        precision.append(p_temp / (at + 1))
        recall.append(p_temp / size)
    print("Precision at", at + 1, ":", sum(precision) / len(precision) * 100, "%")
    print("Recall at", at + 1, ":", sum(recall) / len(recall) * 100, "%")
