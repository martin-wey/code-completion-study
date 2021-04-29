import argparse
import logging
import sys
import os

from tree_sitter import Language, Parser
from tqdm import tqdm

from ast_parser.parser import walk_java, walk_c_sharp


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', type=str, default=None,
        help='Path to file containing the code (one line = one function)'
    )
    parser.add_argument(
        '--lang', type=str, default=None,
        help='Language used to initialize the code_parser.'
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level="DEBUG"
    )

    assert args.lang == 'c_sharp' or args.lang == 'java'

    LANGUAGE = Language('ast_parser/my-languages.so', args.lang)
    parser = Parser()
    parser.set_language(LANGUAGE)

    with open(args.data_path, encoding='utf-8') as f:
        data = f.readlines()

    output = os.path.splitext(args.data_path)
    output_file = f'{output[0]}{output[1]}.flow'

    with open(output_file, encoding='utf-8', mode='w+') as fout:
        for sample in tqdm(data):
            tree = parser.parse(bytes(sample, 'utf-8'))
            if args.lang == 'java':
                code_flow = walk_java(tree, sample)
            else:
                code_flow = walk_c_sharp(tree, sample)
            fout.write(' '.join(code_flow))
            fout.write('\n')


if __name__ == '__main__':
    main()
