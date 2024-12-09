import pytest
from openai.types.beta.threads.runs.run_step import Usage

from agency_swarm.util.tracking import SQLiteTracker


@pytest.fixture
def sqlite_tracker():
    tracker = SQLiteTracker(":memory:")
    yield tracker


def test_sqlite_track_and_get_total_tokens(sqlite_tracker):
    usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    sqlite_tracker.track_usage(
        usage, "test_assistant", "test_thread", "gpt-4o", "sender", "recipient"
    )
    totals = sqlite_tracker.get_total_tokens()
    assert totals == usage


def test_sqlite_multiple_entries(sqlite_tracker):
    # Insert multiple usage entries
    usages = [
        Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        Usage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
    ]
    for u in usages:
        sqlite_tracker.track_usage(
            u, "assistant", "thread", "gpt-4o", "sender", "recipient"
        )

    totals = sqlite_tracker.get_total_tokens()
    assert totals == Usage(prompt_tokens=30, completion_tokens=15, total_tokens=45)
