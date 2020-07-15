import string
import spacy

from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS


class TextPreprocessor:
    def __init__(self):
        self.parser = English()
        self.nlp = spacy.load('en_core_web_sm')

    def tokenize_word(self, sample):
        sample = sample.replace("/", " ").replace(".", " ")

        symbols = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”", "/"]

        tokens = self.nlp(sample)
        lemmas = []
        for tok in tokens:
            lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)

        tokens = lemmas
        tokens = [tok for tok in tokens if tok not in STOP_WORDS]
        tokens = [tok for tok in tokens if tok not in symbols]
        tokens = [tok for tok in tokens if tok != ""]

        return tokens

    def tokenize_sentence(self, text):
        about_txt = self.nlp(text)
        sentences = list(about_txt.sents)

        return sentences


if __name__ == '__main__':
    text_processor = TextPreprocessor()
