import pickle

from src.feature.estimator import FeatureEstimator
from src.preprocess.preprocessor import NLTKPreprocessor
from settings import FORMALITY_MODEL, COEFFICIENT_A, COEFFICIENT_B, COEFFICIENT_C, COEFFICIENT_D, COEFFICIENT_E, \
    COEFFICIENT_F, COEFFICIENT_G


class FormalityCalculator:
    def __init__(self):
        self.feature_extractor = FeatureEstimator()
        self.text_preprocessor = NLTKPreprocessor()
        with open(FORMALITY_MODEL, 'rb') as f_:
            bundle = pickle.load(f_, encoding='latin1')
        self.clf = bundle['clf']
        self.dv = bundle['dv']

    def calculate_formality_text(self, text):
        score = 0
        sentences = self.text_preprocessor.parse(text)['sentences']
        for sentence in sentences:
            features = self.feature_extractor.extract_features(sent=sentence)
            score_dict = self.dv.transform(features)
            score += self.clf.predict(score_dict)[0]
        score = score / len(sentences)
        formality_score = round(COEFFICIENT_D + COEFFICIENT_E * score + COEFFICIENT_F * score ** COEFFICIENT_A -
                                COEFFICIENT_F * score ** COEFFICIENT_B - COEFFICIENT_G * score ** COEFFICIENT_C, 1)

        return formality_score


if __name__ == '__main__':
    FormalityCalculator()
