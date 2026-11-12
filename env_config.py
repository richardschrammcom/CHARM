"""Shared .env loading for all Charm processes.

Precedence is deliberately split:

* **Runtime injections win** for ordinary settings. The dashboard starts each agent
  as a subprocess with `DEFAULT_MODEL` set to that agent's dropdown selection, so a
  blanket `override=True` would let `.env` clobber the per-agent model choice.

* **`.env` wins for provider credentials.** A stale `OPENAI_API_KEY` (or similar)
  exported in whatever shell launched the process would otherwise silently shadow
  the configured key, producing confusing auth/quota errors from the provider while
  the rest of the system looks healthy.
"""

import os
from typing import Union
from pathlib import Path

from dotenv import load_dotenv, dotenv_values

# Credentials that .env owns outright, regardless of the ambient environment.
_CREDENTIAL_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "BROWSER_USE_API_KEY",
)


def load_charm_env(env_path: Union[str, Path]) -> None:
    """Load `.env`, letting runtime injections stand but forcing credentials from file."""
    load_dotenv(env_path)

    values = dotenv_values(env_path)
    for key in _CREDENTIAL_VARS:
        value = values.get(key)
        if value:
            os.environ[key] = value
