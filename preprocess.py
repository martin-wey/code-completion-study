"""
This script can be used to tokenize a line dataset of code.
it uses gigantic-codeprep library that leverages Pygments to parse languages.
"""

import argparse
from pathlib import Path
import os
import json
import logging

import codeprep.api.text as cp
from tqdm import tqdm

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None, required=True,
                        help='Path to the data files to be preprocessed.')
    parser.add_argument('--ext', type=str, default=None, required=True,
                        help='Extension of the files.')
    parser.add_argument('--merge_single_file', type=bool, default=False,
                        help='Output the results in a single file or in separate files.')
    parser.add_argument('--output_path', type=str, default=None,
                        help='If merge_single_file is True, then choose a path to output the results.')
    args = parser.parse_args()

    if args.merge_single_file:
        output_file = args.output_path
        with open(output_file, 'w'): pass

    for path in Path(args.data_dir).rglob(f'*.{args.ext}'):
        if not args.merge_single_file:
            output = os.path.splitext(path)
            output_file = f'{output[0]}_cleaned{output[1]}'
            with open(output_file, 'w'): pass

        logger.info(f'Parsing file: {path}')
        with open(path, encoding='utf-8') as f, open(output_file, encoding='utf-8', mode='a') as f2:
            if args.ext == 'jsonl':
                data = [json.loads(jline) for jline in f.readlines()]
            else:
                data = f.readlines()

            for line in tqdm(data):
                if args.ext == 'jsonl':
                    tokens = cp.nosplit(
                        line['code'],
                        no_com=True,
                        no_spaces=True,
                        return_metadata=True
                    )
                else:
                    tokens = cp.nosplit(line)
                while '<comment>' in tokens:
                    tokens.remove('<comment>')
                f2.write(' '.join(tokens))
                f2.write('\n')


if __name__ == '__main__':
    main()
