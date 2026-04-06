"""Router configuration and structured logging setup.

Uses a frozen dataclass for immutable config — intentionally different from
env-loading patterns. CLI args override defaults at runtime.
"""

import logging
import os
import sys
from dataclasses import dataclass

import structlog


@dataclass(frozen=True)
class RouterConfig:
    simple_token_threshold: int = 200
    complex_token_threshold: int = 1000
    default_output_format: str = "markdown"
    target_simple: str = "gpt-4o-mini"
    target_medium: str = "claude-3.5-haiku"


_logging_configured = False


def setup_logging(verbose: bool = False):
    global _logging_configured
    if _logging_configured:
        return

    level_name = os.environ.get("LLM_ROUTER_LOG_LEVEL", "WARNING").upper()
    level = getattr(logging, level_name, logging.WARNING)
    if verbose:
        level = logging.DEBUG

    logging.basicConfig(
        format="%(message)s",
        level=level,
        stream=sys.stderr,
        force=True,
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
    _logging_configured = True
