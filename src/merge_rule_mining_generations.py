from sys import argv


GENERATION_FILENAME = argv[1]
OUTPUT_FILENAME = argv[2]


res = dict()
with open(GENERATION_FILENAME) as f:
    for line in f:
        line = line.strip().split("\t")
        if len(line) != 5:
            print("Problem with:", line)
            continue
        spo = ",".join(line[:3])
        if spo not in res:
            res[spo] = []
        res[spo].append(float(line[3]) * float(line[4]))


KB = []
for spo, scores in res.items():
    KB.append((spo, sum(scores)))


KB = [x[0] + "\t" + str(x[1]) for x in sorted(KB, key=lambda x: -x[1])]

with open(OUTPUT_FILENAME, "w") as f:
    f.write("triple\tfrequency\n")
    f.write("\n".join(KB) + "\n")

