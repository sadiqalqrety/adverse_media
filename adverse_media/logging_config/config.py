"""Structured JSON logging configuration.

Call ``configure_logging()`` once at application start-up (done by app.py).
The log level is read from the ``LOG_LEVEL`` environment variable; the
default is ``INFO``.

Every log record is emitted as a single JSON object so that log-aggregation
systems (Datadog, Splunk, CloudWatch Logs Insights, etc.) can query on
individual fields without needing a regex parser.

Example query (CloudWatch Logs Insights):
    fields @timestamp, level, event, name, dob, match_assessment, sentiment
    | filter match_assessment = "LIKELY_MATCH"
    | filter sentiment = "NEGATIVE"
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone

LOG_LEVEL_ENV = "LOG_LEVEL"
_DEFAULT_LEVEL = "INFO"

# Attributes present on every LogRecord — excluded from the structured extras
# so they are not double-emitted.
_STANDARD_RECORD_ATTRS = frozenset({
    "args", "created", "exc_info", "exc_text", "filename", "funcName",
    "id", "levelname", "levelno", "lineno", "message", "module", "msecs",
    "msg", "name", "pathname", "process", "processName", "relativeCreated",
    "stack_info", "thread", "threadName", "taskName",
})


class JsonFormatter(logging.Formatter):
    """Formats every log record as a single-line JSON object.

    Standard fields always present:
        timestamp  — ISO-8601 UTC
        level      — DEBUG / INFO / WARNING / ERROR / CRITICAL
        logger     — dotted logger name
        event      — the log message string

    Any keyword passed via ``extra={...}`` is merged into the top-level object,
    making fields directly queryable without nested path syntax.
    """

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()

        payload: dict = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.message,
        }

        # Merge caller-supplied extra fields
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS:
                payload[key] = value

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging() -> None:
    """Configure the ``adverse_media`` logger from the environment.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    logger = logging.getLogger("adverse_media")

    # NullHandler is added by __init__.py as a library default; don't treat it
    # as "already configured" — replace it with the real handler below.
    real_handlers = [h for h in logger.handlers if not isinstance(h, logging.NullHandler)]
    if real_handlers:
        return  # already configured with a real handler
    logger.handlers = []  # remove the NullHandler

    level_name = os.environ.get(LOG_LEVEL_ENV, _DEFAULT_LEVEL).upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())

    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False
