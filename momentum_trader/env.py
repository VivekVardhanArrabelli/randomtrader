"""Environment configuration helpers."""

from __future__ import annotations

import os


class MissingEnvironment(Exception):
    pass


def get_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        raise MissingEnvironment(f"Missing required environment variable: {name}")
    return value
