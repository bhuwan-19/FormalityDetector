import argparse

from src.formality.calculator import FormalityCalculator

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', action='store', help='A text string')
    text = parser.parse_args().title

    formality_score = FormalityCalculator().calculate_formality_text(text=text)

    print(f"Text Formality:{formality_score}")
