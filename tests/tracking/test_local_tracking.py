import json
from unittest.mock import MagicMock

import pytest
from openai.types.beta.threads.runs.run_step import Usage

from agency_swarm.util.tracking import LocalTracker


@pytest.fixture
def local_tracker():
    tracker = LocalTracker(":memory:")
    yield tracker


def test_local_track_assistant_message(local_tracker):
    # Mock OpenAI client and responses
    mock_client = MagicMock()
    mock_message_log = MagicMock()
    mock_message_log.data = [
        MagicMock(role="user", content=[MagicMock(text=MagicMock(value="Hello"))]),
        MagicMock(
            role="assistant", content=[MagicMock(text=MagicMock(value="Hi there"))]
        ),
    ]
    mock_client.beta.threads.messages.list.return_value = mock_message_log

    mock_run = MagicMock()
    mock_run.model = "gpt-4o"
    mock_run.usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    mock_run.assistant_id = "test_assistant"
    mock_client.beta.threads.runs.retrieve.return_value = mock_run

    # Track a message
    local_tracker.track_assistant_message(
        client=mock_client,
        thread_id="test_thread",
        run_id="test_run",
        message_content="Test response",
        sender_agent_name="sender",
        recipient_agent_name="recipient",
    )

    # Verify the message was stored
    cursor = local_tracker.conn.cursor()
    # Use explicit column names in the query
    cursor.execute("""
        SELECT assistant_id, thread_id, run_id, message_content,
               input_messages, model, prompt_tokens, completion_tokens,
               total_tokens, sender_agent_name, recipient_agent_name
        FROM usage_tracking
        WHERE run_id = 'test_run'
    """)
    row = cursor.fetchone()

    assert row is not None
    # Use named indices to make the test more maintainable
    assistant_id, thread_id, run_id, message_content, input_messages, model, *_ = row

    assert thread_id == "test_thread"
    assert run_id == "test_run"
    assert model == "gpt-4o"
    assert message_content == "Test response"

    # Verify input messages were stored as JSON
    input_messages_dict = json.loads(input_messages)
    assert len(input_messages_dict) == 1
    assert input_messages_dict[0]["role"] == "user"
    assert input_messages_dict[0]["content"] == "Hello"
