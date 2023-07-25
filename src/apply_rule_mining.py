import pandas as pd
from sys import argv
import sys

from rule import Rule

KB = argv[1]
AMIE_OUTPUT = argv[2]
OUTPUT = argv[3]


amie_results = pd.read_csv(AMIE_OUTPUT, sep="\t")
amie_results = amie_results.sort_values(by="Std Confidence", ascending=False)[['Rule', 'Std Confidence']]
amie_results = amie_results[amie_results['Std Confidence'] > 0.5]


print("TOP RULES")
print("\n".join(rule + "\t" + str(score) for rule, score in amie_results.to_numpy()[:10]))

rules = [(Rule(rule), score) for rule, score in amie_results.to_numpy()]
rules = [x for x in rules if x[0].is_valid()]

print("Number of rules:", len(rules))


res = []
with open(KB) as f:
    for i, line in enumerate(f):
        if i%1000 == 0 and res:
            sys.stdout.write("\033[K")
            print(i, len(res), res[-1], end="\r")
        line = line.strip().split("\t")
        if len(line) != 4:
            continue
        for rule, score in rules:
            trans = rule.apply(line[0], line[1], line[2])
            if trans is not None:
                res.append("\t".join(trans) + "\t" + str(score) + "\t" + line[3])


print("Number of translations:", len(res))
with open(OUTPUT, "w") as f:
    f.write("\n".join(res))
