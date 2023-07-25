''' Computes a manual mapping of an OpenKB

It takes three arguments:
    * The OpenKB, formatted as subject, predicate, object, score and separated by a tab.
    * The mapping file. It must be a TSV file of two columns. The first one represents the open relation, and the second
    one the translation
    * The output filename

'''

import sys

from predicate_conceptnet import predicate_mapping

KB_FILENAME = sys.argv[1]
MAPPING_FILENAME = sys.argv[2]
OUTPUT_FILENAME = sys.argv[3]
OUTPUT_GENERATION_FILENAME = OUTPUT_FILENAME + ".generations"

DEFAULT_TO_CAPABLEOF = False

MAPPING = dict()
with open(MAPPING_FILENAME) as f:
    for line in f:
        line = line.strip().split("\t")
        if len(line) >= 2:
            MAPPING[line[0]] = line[1:]


TRANSLATION = dict()
GENERATIONS = []

with open(KB_FILENAME) as f:
    for line in f:
        line = line.strip().split("\t")
        if len(line) < 4:
            continue
        s = line[0]
        p = line[1]
        o = line[2]
        try:
            score = float(line[3])
        except ValueError:
            continue
        all_gens = []
        if p in MAPPING:
            for p_trans in MAPPING[p]:
                new_p = p_trans
                new_o = o
                new_s = s
                if p_trans == "None" or not p_trans:
                    new_p = "CapableOf"
                    new_o = p + " " + o
                elif p_trans[-1] == "-":
                    new_p = p_trans[:-1]
                    new_s = o
                    new_o = s
                temp = new_s + "," + new_p + "," + new_o
                if temp not in TRANSLATION:
                    TRANSLATION[temp] = []
                TRANSLATION[temp].append(score)
                all_gens.append(new_s + "," + predicate_mapping.get(new_p, "") + "," + new_o)
        elif DEFAULT_TO_CAPABLEOF:
            o = p + " " + o
            p = "CapableOf"
            temp = s + "," + p + "," + o
            if temp not in TRANSLATION:
                TRANSLATION[temp] = []
            TRANSLATION[temp].append(score)
            all_gens.append(new_s + "," + predicate_mapping.get(new_p, "") + "," + new_o)
        GENERATIONS.append(s + "," + p + "," + o + "\t" + "\t".join(all_gens) + "\n")


for key, value in TRANSLATION.items():
    new_value = max(x / (i + 1) for i, x in enumerate(value))
    TRANSLATION[key] = new_value


TRANSLATION_RANKED = [y[0] + "\t" + str(y[1]) for y in sorted(TRANSLATION.items(), key=lambda x: -x[1])]


with open(OUTPUT_FILENAME, "w") as f:
    f.write("\n".join(TRANSLATION_RANKED) + "\n")

with open(OUTPUT_GENERATION_FILENAME, "w") as f:
    f.write("".join(GENERATIONS))
