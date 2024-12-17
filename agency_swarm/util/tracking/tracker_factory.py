from typing import Literal

from agency_swarm.util.tracking.langfuse_tracker import LangfuseTracker
from agency_swarm.util.tracking.local_tracker import LocalTracker


def get_tracker_by_name(name: Literal["local", "langfuse"] = "local"):
    if name == "langfuse":
        return LangfuseTracker()
    return LocalTracker()
