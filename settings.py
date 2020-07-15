import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(CUR_DIR, 'utils', 'model')
FORMALITY_MODEL = os.path.join(MODEL_DIR, 'answers.light.clf')
WORD2VEC_MODEL = os.path.join(MODEL_DIR, 'GoogleNews-vectors-negative300.bin')
