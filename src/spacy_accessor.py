import spacy


SEPARATOR = "\n\n"
LEN_SEPARATOR = len(SEPARATOR)


class SpacyAccessor(object):

    def __init__(self, model="en_core_web_sm"):
        if model is None:
            model = "en_core_web_sm"
        self._nlp = spacy.load(model)

    def lemmatize(self, sentence):
        tokens = self._nlp(sentence)
        return [x.lemma_ for x in tokens]

    def annotate(self, sentence):
        return self._nlp(sentence)


spacy_annotator = None


def get_default_annotator():
    global spacy_annotator
    if spacy_annotator is None:
        spacy_annotator = SpacyAccessor()
    return spacy_annotator
