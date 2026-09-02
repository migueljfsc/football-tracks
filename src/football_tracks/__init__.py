"""Broadcast football clip -> player tracks in pitch metres.

The pipeline runs as independent stages, each reading the previous stage's artefact
from work/<clip>/ and writing its own. See PLAN.md.
"""

__version__ = "0.3.1"
