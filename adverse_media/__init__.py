"""adverse_media — screening pipeline package."""

import logging

from .checker import AdverseMediaChecker

__all__ = ["AdverseMediaChecker"]

# Library best practice: attach a NullHandler so log records are silently
# discarded unless the host application configures a real handler.
logging.getLogger(__name__).addHandler(logging.NullHandler())
