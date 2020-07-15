import pickle

from src.feature.estimator import FeatureEstimator
from src.preprocess.tokenizer import TextPreprocessor
from settings import FORMALITY_MODEL


class FormalityCalculator:
    def __init__(self):
        self.feature_extractor = FeatureEstimator()
        self.text_preprocessor = TextPreprocessor()
        with open(FORMALITY_MODEL, 'rb') as f_:
            bundle = pickle.load(f_, encoding='latin1')
        self.clf = bundle['clf']
        self.dv = bundle['dv']

    def calculate_formality_text(self, text):

        score = 0
        sentences = self.text_preprocessor.tokenize_sentence(text=text)
        for sentence in sentences:
            features = self.feature_extractor.extract_features(sent=sentence)
            score_dict = self.dv.transform(features)
            score += self.clf.predict(score_dict)[0]
        score = score / len(sentences)

        return score


if __name__ == '__main__':
    FormalityCalculator()
