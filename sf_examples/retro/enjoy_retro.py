import sys

from sample_factory.enjoy import enjoy
from sf_examples.retro.train_retro import parse_retro_args, register_retro_components


def main():
    """Script entry point."""
    register_retro_components()
    cfg = parse_retro_args(evaluation=True)

    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
