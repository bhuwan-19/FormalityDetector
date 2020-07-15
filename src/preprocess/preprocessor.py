"""
Wrapper classes to allow switching between different NLP toolkits for preprocessing, tagging, parsing, etc
"""
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet


class NLTKPreprocessor(object):
    """Use NLTK to return a bundle in the same format as the one CoreNLP returns"""

    def __init__(self):
        from nltk.stem.wordnet import WordNetLemmatizer
        self.lemmatizer = WordNetLemmatizer()

    @staticmethod
    def _get_word_net_pos(tag):

        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN  # this is the default when POS is not known

    def parse(self, document):

        data = {'sentences': []}
        for sent in sent_tokenize(document):
            tokens = [w for w in word_tokenize(sent)]
            postags = [t for w, t in pos_tag(tokens)]
            lemmas = [self.lemmatizer.lemmatize(w, self._get_word_net_pos(t)) for w, t in zip(tokens, postags)]
            data['sentences'].append({'tokens': tokens, 'lemmas': lemmas, 'pos': postags})
        return data
