from sys import argv


IN_FILENAME = argv[1]
OUT_FILENAME = argv[2]


res = dict()
with open(IN_FILENAME) as f:
    for line in f:
        line = line.strip()
        if "[SEP]" in line:
            line = line.split("[SEP]")[1]
        s, p, o, r = line.split("\t")
        key = s + "," + p + "," + o
        if key not in res:
            res[key] = []
        res[key].append(float(r))


KB = [(x, sum(y)) for x, y in res.items()]
KB = [x + "\t" + str(y) + "\n" for x, y in sorted(KB, key=lambda x: -x[1])]


with open(OUT_FILENAME, "w") as f:
    f.write("".join(KB))
