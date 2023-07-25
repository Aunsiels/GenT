from sys import argv

KB = argv[1]
TEST_SET_FILENAME = argv[2]
OUTPUT_FILNAME = argv[3]


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
with open(KB) as f:
    for line_full in f:
        line = line_full.strip().split("\t")
        if len(line) < 3:
            continue
        spo = "\t".join(line[:3])
        if spo in goldstandard:
            res.append(line_full)

with open(OUTPUT_FILNAME, "w") as f:
    f.write("".join(res))
