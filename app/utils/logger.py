"""structlog configuration and logger access."""

import logging
import sys

import structlog

_configured = False


def setup_logging(debug: bool = False) -> None:
    """Configure structlog for the whole application.

    Args:
        debug: When True, use DEBUG level and a pretty console renderer;
            otherwise use INFO level and JSON output (production friendly).
    """
    global _configured
    if _configured:
        return

    level = logging.DEBUG if debug else logging.INFO
    renderer = (
        structlog.dev.ConsoleRenderer()
        if debug
        else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso", utc=False),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Keep third-party stdlib logging at the same level.
    logging.basicConfig(level=level, stream=sys.stdout)

    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger.

    Args:
        name: Optional logger name, typically `__name__` of the caller.
    """
    return structlog.get_logger(name)
