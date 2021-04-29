"""
Train a SentencePiece tokenizer with BPE.
"""

import os
import argparse

from tokenizers.implementations import ByteLevelBPETokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_file', type=str,
        help='The input training data file (a text file).'
    )
    parser.add_argument(
        '--output_dir', type=str,
        help='Path where the tokenizer should be saved.'
    )
    parser.add_argument(
        '--vocab_size', type=int, default=52000,
        help='Size of the vocabulary (=number of merges)'
    )
    parser.add_argument(
        '--min_frequency', type=int, default=2,
        help='Minimum frequency of a token.'
    )
    args = parser.parse_args()

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[args.train_file],
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        show_progress=True
    )

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    tokenizer.save_model(args.output_dir)


if __name__ == '__main__':
    main()