"""
Make the alignment between two knowledge bases

It takes three arguments:
    * The KB we want to predict, as a tsv
    * The KB we want to translate, as a tsv
    * The output file
"""

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sys import argv


source_filename = argv[1]
target_filename = argv[2]
save_filename = argv[3]


SW = stopwords.words('english')
lemmatizer = WordNetLemmatizer()


def lemmatize(text):
    tokens = word_tokenize(text)
    res = []
    for token in tokens:
        if token not in SW:
            res.append(lemmatizer.lemmatize(token))
    return " ".join(res)


conceptnet_so = dict()


with open(source_filename) as f:
    for line in f:
        line = line.strip().replace("_", " ").split("\t")
        s = line[0]
        p = line[1]
        o = line[2]
        o_lemm = lemmatize(o)
        if not o_lemm:
            continue
        if s not in conceptnet_so:
            conceptnet_so[s] = dict()
        if o_lemm not in conceptnet_so[s]:
            conceptnet_so[s][o_lemm] = set()
        conceptnet_so[s][o_lemm].add((s, p, o))


res = []


with open(target_filename) as f:
    for line in f:
        line = line.strip().split("\t")
        s = line[0]
        p = line[1]
        o = line[2]
        p_lemm = lemmatize(p)
        o_lemm = lemmatize(o)
        long_o = p + " " + o
        long_o_lemm = p_lemm + " " + o_lemm
        if s in conceptnet_so:
            if o_lemm in conceptnet_so[s]:
                for s_cn, p_cn, o_cn in conceptnet_so[s][o_lemm]:
                    res.append(s + "\t" + p + "\t" + o + "[SEP]" + s_cn + "\t" + p_cn + "\t" + o_cn)
            if long_o_lemm in conceptnet_so[s]:
                for s_cn, p_cn, o_cn in conceptnet_so[s][long_o_lemm]:
                    res.append(s + "\t" + p + "\t" + o + "[SEP]" + s_cn + "\t" + p_cn + "\t" + o_cn)
        if o_lemm in conceptnet_so:
            if s in conceptnet_so[o_lemm]:
                for s_cn, p_cn, o_cn in conceptnet_so[o_lemm][s]:
                    res.append(s + "\t" + p + "\t" + o + "[SEP]" + s_cn + "\t" + p_cn + "\t" + o_cn)
        if long_o_lemm in conceptnet_so:
            if s in conceptnet_so[long_o_lemm]:
                for s_cn, p_cn, o_cn in conceptnet_so[long_o_lemm][s]:
                    res.append(s + "\t" + p + "\t" + o + "[SEP]" + s_cn + "\t" + p_cn + "\t" + o_cn)


cn_rel = set()

for x in res:
    cn_rel.add(x.split("\t")[3])

res_trans = dict()

for x in cn_rel:
    temp = ""
    for c in x:
        if c.isupper() and not temp:
            temp += c.lower()
        elif c.isupper():
            temp += " " + c.lower()
        else:
            temp += c
    res_trans[x] = temp


res2 = []

for x in res:
    x = x.replace("_", " ")
    x_s = x.split("[SEP]")
    if x_s[0] == x_s[1]:
        continue
    for key, value in res_trans.items():
        x = x.replace(key, value)
    res2.append(x)


with open(save_filename, "w") as f:
    f.write("\n".join(res2))


