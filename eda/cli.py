import argparse
import logging
import yaml

from eda.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_pipeline(config)


if __name__ == "__main__":
    main()