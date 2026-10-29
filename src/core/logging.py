"""One place to set up logging, so every module prints the same readable format.

Format: time | LEVEL | module name | message
The level (DEBUG/INFO/...) comes from config.yaml, so you can get more detail
without editing code.
"""
import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO") -> None:
    # force=True replaces any earlier setup, so calling this twice is safe.
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=_FORMAT,
        stream=sys.stdout,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Each module calls get_logger(__name__) so log lines show where they came from."""
    return logging.getLogger(name)
