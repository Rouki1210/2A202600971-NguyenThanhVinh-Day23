from types import SimpleNamespace

from langgraph_agent_lab.persistence import _resolve_database_url, state_history_evidence, thread_config


def test_thread_config_uses_langgraph_configurable_thread_id():
    assert thread_config("thread-S01") == {"configurable": {"thread_id": "thread-S01"}}


def test_database_url_can_be_resolved_from_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/langgraph_lab")
    assert (
        _resolve_database_url("env:DATABASE_URL")
        == "postgresql://postgres:postgres@localhost:5432/langgraph_lab"
    )


def test_state_history_evidence_summarizes_snapshots():
    class FakeGraph:
        def get_state_history(self, config):
            assert config == thread_config("thread-S01")
            return iter(
                [
                    SimpleNamespace(
                        values={
                            "route": "simple",
                            "attempt": 0,
                            "final_answer": "ok",
                            "pending_question": None,
                            "events": [{"node": "finalize"}],
                        },
                        metadata={"checkpoint_id": "checkpoint-1"},
                    )
                ]
            )

    evidence = state_history_evidence(FakeGraph(), "thread-S01")
    assert evidence["thread_id"] == "thread-S01"
    assert evidence["has_checkpoint_history"] is True
    assert evidence["history_count"] == 1
    assert evidence["snapshots"][0]["checkpoint_id"] == "checkpoint-1"
