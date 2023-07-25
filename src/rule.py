from nltk.corpus import wordnet as wn


class Rule:

    def __init__(self, rule_str):
        body, head = rule_str.split(" => ")
        self.head_relation = head.split("  ")[1]
        body_split = body.strip().split("  ")
        self.inverse_so = False
        self.words_in_predicates = None
        self.type_subject = None
        self.type_object = None
        for i in range(0, len(body_split), 3):
            s = body_split[i]
            p = body_split[i + 1]
            o = body_split[i + 2]
            if p == "INSUBJ":
                if o == "?b":
                    self.inverse_so = True
            elif p == "INOBJ":
                if o == "?a":
                    self.inverse_so = True
            elif p == "ISA":
                if s == "?a":
                    self.type_subject = o
                elif s == "?b":
                    self.type_object = o
            elif p == "CONTAINS":
                self.words_in_predicates = o

    def is_valid(self):
        return self.words_in_predicates is not None \
            or self.type_subject is not None \
            or self.type_object is not None

    def apply(self, s, p, o):
        if self.words_in_predicates is not None:
            if self.words_in_predicates not in p.split(" "):
                return None
        if self.type_subject is not None:
            hypers = get_all_hypernyms(s)
            if self.type_subject not in hypers:
                return None
        if self.type_object is not None:
            hypers = get_all_hypernyms(o)
            if self.type_object not in hypers:
                return None
        if self.inverse_so:
            return o, self.head_relation, s
        return s, self.head_relation, o


def get_all_hypernyms(word):
    res = set()
    synsets = wn.synsets(word, "n")
    to_process = []
    for synset in synsets:
        for hyper in synset.hypernyms():
            to_process.append(hyper)
    while to_process:
        current = to_process.pop()
        res.add(current.name())
        for hyper in current.hypernyms():
            to_process.append(hyper)
    return res
