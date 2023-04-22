import argparse
from scripts import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--feature", required=False)
    parser.add_argument("--label", required=False)
    parser.parse_args()


if __name__ == '__main__':
    main()
