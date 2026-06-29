"""Checkpointer adapter and persistence evidence helpers."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any


_DOTENV_LOADED = False


def _load_dotenv() -> None:
    """Load simple KEY=VALUE pairs so DATABASE_URL works from a local .env file."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True

    env_path = Path(".env")
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _resolve_database_url(database_url: str | None) -> str | None:
    _load_dotenv()
    if database_url and database_url.startswith("env:"):
        return os.getenv(database_url.removeprefix("env:"))
    return database_url or os.getenv("DATABASE_URL")


def _sqlite_path(database_url: str | None) -> Path:
    resolved_url = _resolve_database_url(database_url)
    if not resolved_url:
        return Path("outputs/checkpoints.sqlite")
    if resolved_url.startswith("sqlite:///"):
        return Path(resolved_url.removeprefix("sqlite:///"))
    if "://" in resolved_url:
        raise ValueError(f"SQLite checkpointer cannot use database_url={resolved_url!r}")
    return Path(resolved_url)


def _setup_checkpointer(checkpointer: Any) -> Any:
    setup = getattr(checkpointer, "setup", None)
    if callable(setup):
        setup()
    return checkpointer


def thread_config(thread_id: str) -> dict[str, dict[str, str]]:
    """Build the LangGraph run config that binds checkpoints to one thread."""
    return {"configurable": {"thread_id": thread_id}}


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:
    """Return a LangGraph checkpointer.

    TODO(student): implement SQLite support for the persistence extension track.
    The starter provides MemorySaver only — SQLite/Postgres are extension tasks.

    For SQLite:
    - pip install langgraph-checkpoint-sqlite
    - Use SqliteSaver with sqlite3.connect() and WAL mode
    - See: https://langchain-ai.github.io/langgraph/how-tos/persistence/
    """
    if kind == "none":
        return None
    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if kind == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise RuntimeError(
                "SQLite checkpointing requires: pip install langgraph-checkpoint-sqlite"
            ) from exc

        db_path = _sqlite_path(database_url)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return _setup_checkpointer(SqliteSaver(conn=conn))
    if kind == "postgres":
        database_url = _resolve_database_url(database_url)
        if not database_url:
            raise RuntimeError(
                "Postgres checkpointing requires database_url or DATABASE_URL in .env"
            )
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
        except ImportError as exc:
            raise RuntimeError(
                "Postgres checkpointing requires: pip install langgraph-checkpoint-postgres"
            ) from exc

        manager = PostgresSaver.from_conn_string(database_url)
        checkpointer = manager.__enter__()
        setattr(checkpointer, "_context_manager", manager)
        return _setup_checkpointer(checkpointer)
    raise ValueError(f"Unknown checkpointer kind: {kind}")


def state_history_evidence(graph: Any, thread_id: str, limit: int = 10) -> dict[str, Any]:
    """Summarize persisted state history for demo/report evidence.

    Pass the compiled graph returned by build_graph(checkpointer=...) and the
    thread_id from initial_state(). This works with MemorySaver and SQLite
    checkpointers when the graph was invoked with the same thread_id config.
    """
    snapshots: list[dict[str, Any]] = []
    config = thread_config(thread_id)

    for index, snapshot in enumerate(graph.get_state_history(config)):
        if index >= limit:
            break
        values = getattr(snapshot, "values", {}) or {}
        metadata = getattr(snapshot, "metadata", {}) or {}
        snapshots.append(
            {
                "index": index,
                "route": values.get("route"),
                "attempt": values.get("attempt"),
                "final_answer_present": bool(values.get("final_answer")),
                "pending_question_present": bool(values.get("pending_question")),
                "events_count": len(values.get("events", []) or []),
                "checkpoint_id": metadata.get("checkpoint_id")
                or metadata.get("thread_ts")
                or metadata.get("step"),
            }
        )

    return {
        "thread_id": thread_id,
        "history_count": len(snapshots),
        "has_checkpoint_history": bool(snapshots),
        "snapshots": snapshots,
    }
