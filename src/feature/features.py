import sys
import numpy as np

from gensim.models import KeyedVectors
from nltk.corpus import cmudict
from nltk.util import ngrams
from settings import WORD2VEC_MODEL, CUR_DIR


class FeatureGroup(object):
    """
    A FeatureGroup is a glorified dictionary representing a set of similar features
    """

    def __init__(self, name, homedir):
        self.name = name
        self.homedir = homedir

    def f_name(self, f):
        return '%s_%s' % (self.name, f)

    def add(self, features, feature, value):
        features[self.f_name(feature)] = float(value)
        return features

    def inc(self, features, feature, value):
        fnm = self.f_name(feature)
        value = float(value)
        if fnm in features:
            features[fnm] += value
        else:
            self.add(features, feature, value)
        return features

    def norm_by(self, features, d, exclude=None):
        if exclude is None:
            exclude = []
        exclude = [self.f_name(f) for f in exclude]
        d = float(d)
        return {f: v / d for f, v in features.iteritems() if f not in exclude}

    @staticmethod
    def feature_string(features):
        return ' '.join(['%s:%s' % (f, v) for f, v in features.iteritems()])


class NgramFeatures(FeatureGroup):
    """
    Feature group for ngram features
    """

    def __init__(self, name, homedir, n=3):
        super(NgramFeatures, self).__init__(name, homedir)
        self.N = n

    def extract(self, doc):

        features = {}
        for sent in doc:
            words = sent['tokens']
            for n in range(1, self.N):
                for ngm in ngrams(words, n):
                    features = self.add(features, ' '.join([w.lower() for w in ngm]), 1)
        return features


class CaseFeatures(FeatureGroup):
    """
    Feature group for casing and punctuation features
    """

    def extract(self, doc):
        features = {}
        for sent in doc:
            # Casing
            caps = sum([1 if (w.isupper() and not (w == 'I')) else 0 for w in sent['tokens']])
            features = self.inc(features, 'Number of capitalized words', caps)
            all_lower = sum([0 if w.islower() else 1 for w in sent['tokens'] if w.isalpha()]) > 0
            features = self.inc(features, 'All lowercase sentence', all_lower)
            features = self.inc(features, 'Lowercase initial sentence', 1 if sent['tokens'][0].islower() else 0)

            # Punctuation
            for punctuation in ['!', '?', '...']:
                for w in sent['tokens']:
                    if punctuation in w:
                        features = self.inc(features, 'Number of %s per sentence' % punctuation, 1)

        return features


class ReadabilityFeatures(FeatureGroup):
    """
    Feature group for length and readability features
    """

    def __init__(self, name, homedir):

        super(ReadabilityFeatures, self).__init__(name, homedir)
        self.d = cmudict.dict()

    # from https://groups.google.com/forum/#!topic/nltk-users/mCOh_u7V8_I
    def _nsyl(self, word):
        word = word.lower()
        if word in self.d:
            return min([len(list(y for y in x if y[-1].isdigit())) for x in self.d[word.lower()]])
        else:
            return 0

    def _fk(self, tokens):
        words = 0
        sentences = 1
        syllables = 0
        for w in tokens:
            words += 1
            syllables += self._nsyl(w)
        if words > 0 and sentences > 0:
            return (0.39 * (words / sentences)) + (11.8 * (syllables / words)) - 15.59
        return 0

    def extract(self, doc):
        features = {}
        all_tokens = []
        for sent in doc:
            all_tokens += sent['tokens']
            features = self.inc(features, 'length in words', len(sent['tokens']))
            features = self.inc(features, 'length in characters', len(' '.join(sent['tokens'])))
        features = self.add(features, 'FK score', self._fk(all_tokens))
        return features


class W2VFeatures(FeatureGroup):
    """
    Feature group for w2v features
    """

    def __init__(self, name, homedir):

        super(W2VFeatures, self).__init__(name, homedir)
        sys.stderr.write("Loading w2v...")
        self.w2v = KeyedVectors.load_word2vec_format(WORD2VEC_MODEL, binary=True)
        sys.stderr.write("done\n")

    def extract(self, doc):

        features = {}
        v = None
        total = 0.
        for sent in doc:
            for w in sent['tokens']:
                try:
                    wv = np.array(self.w2v[w.lower()])
                    if (max(wv) < float('inf')) and (min(wv) > -float('inf')):
                        if v is None:
                            v = wv
                        else:
                            v += wv
                        total += 1
                except KeyError:
                    continue
            if v is not None:
                v = v / total
                for i, n in enumerate(v):
                    if n == float('inf'):
                        n = sys.float_info.max
                    if n == -float('inf'):
                        n = -sys.float_info.max
                    features = self.add(features, 'w2v-%d' % i, n)
            else:
                features = self.add(features, 'w2v-NA', 1)
        return features


if __name__ == '__main__':
    NgramFeatures(name="", homedir=CUR_DIR)
