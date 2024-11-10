import argparse
from core.module.build import Builder

parser = argparse.ArgumentParser(description='Searching paper for User Instruction from Google Scholar & Arxiv')
parser.add_argument('-i', '--input', type=str,  help='User Instruction')

if __name__ == '__main__':
    args = parser.parse_args()

    module = Builder()
    graph = module.compile()
    module.run(graph=graph, instruction=args.input)