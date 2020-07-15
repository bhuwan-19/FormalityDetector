import sys

from src.preprocess.preprocessor import NLTKPreprocessor
from src.feature.features import CaseFeatures, ReadabilityFeatures, W2VFeatures, NgramFeatures
from settings import CUR_DIR


class FeatureEstimator:
    """
    The Featurizer class is responsible for extracting features from a blob of text.
    It is designed under the assumption that the blob of text represents a sentence, but this not necessary.
    Note the performance might degrade if the code is running using larger (or smaller) untils of text.
    """

    def __init__(self):

        self.homedir = CUR_DIR
        self.light_feature_names = ['ngram', 'case', 'readability', 'w2v']
        sys.stderr.write("Initializing featurizer\n")
        use = set(self.light_feature_names)
        self.use_features = self._get_feature_to_use(use)
        self.preprocessor = NLTKPreprocessor()

    def _get_feature_to_use(self, use):
        use_features = []
        if 'case' in use:
            use_features.append(CaseFeatures('case', homedir=self.homedir))
        if 'ngram' in use:
            use_features.append(NgramFeatures('ngram', homedir=self.homedir))
        if 'readability' in use:
            use_features.append(ReadabilityFeatures('readability', homedir=self.homedir))
        if 'w2v' in use:
            use_features.append(W2VFeatures('w2v', homedir=self.homedir))
        return use_features

    def extract_features(self, sent):

        sent = self.preprocessor.parse(sent)['sentences']
        features = {}
        for feats in self.use_features:
            f = feats.extract(sent)
            features.update(f)

        return features


if __name__ == '__main__':
    FeatureEstimator().extract_features(sent="")
