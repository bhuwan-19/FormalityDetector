import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(CUR_DIR, 'utils', 'model')
FORMALITY_MODEL = os.path.join(MODEL_DIR, 'formality.clf')
WORD2VEC_MODEL = os.path.join(MODEL_DIR, 'GoogleNews-vectors-negative300.bin')

COEFFICIENT_A = 2.0
COEFFICIENT_B = 3.0
COEFFICIENT_C = 4.0
COEFFICIENT_D = 2.45
COEFFICIENT_E = 1.22
COEFFICIENT_F = 0.05
COEFFICIENT_G = 0.005
