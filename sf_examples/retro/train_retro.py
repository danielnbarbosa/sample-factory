import sys

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sf_examples.retro.retro_params import retro_override_defaults
from sf_examples.retro.retro_utils import RETRO_ENVS, make_retro_env


def register_retro_envs():
    for env in RETRO_ENVS:
        register_env(env.name, make_retro_env)


def register_retro_components():
    register_retro_envs()


def parse_retro_args(argv=None, evaluation=False):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    retro_override_defaults(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def main():  # pragma: no cover
    """Script entry point."""
    register_retro_components()
    cfg = parse_retro_args()
    status = run_rl(cfg)
    return status


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
