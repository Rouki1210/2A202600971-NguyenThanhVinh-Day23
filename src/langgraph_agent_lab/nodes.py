"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, ApprovalDecision, Route, make_event


class ClassificationResult(BaseModel):
    route: Literal["simple", "tool", "missing_info", "risky", "error"] = Field(
        description="The best workflow route for the support ticket."
    )
    rationale: str = Field(description="Short reason for the classification.")


def _message_text(response: object) -> str:
    """Extract content from a LangChain message or any plain response object."""
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return " ".join(str(item) for item in content)
    return str(content)


def _fallback_classify(query: str) -> ClassificationResult:
    """Deterministic backup used only when no LLM provider is configured locally."""
    text = query.lower()
    risky_terms = ("refund", "delete", "cancel", "send confirmation", "email", "chargeback")
    tool_terms = ("lookup", "order", "status", "tracking", "search", "find")
    missing_terms = ("fix it", "help me", "it is broken", "not working", "can you fix")
    error_terms = ("timeout", "failure", "crash", "unavailable", "cannot recover", "system failure")
    if any(term in text for term in risky_terms):
        return ClassificationResult(route="risky", rationale="Side-effecting customer action.")
    if any(term in text for term in tool_terms):
        return ClassificationResult(route="tool", rationale="Requires information lookup.")
    if any(term in text for term in missing_terms) or len(text.split()) <= 4:
        return ClassificationResult(route="missing_info", rationale="The request lacks enough detail.")
    if any(term in text for term in error_terms):
        return ClassificationResult(route="error", rationale="System failure or transient error.")
    return ClassificationResult(route="simple", rationale="General support question.")


def _risk_level(route: str) -> str:
    return "high" if route == Route.RISKY.value else "low"


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── TODO(student): implement ALL nodes below ────────────────────────


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM.

    *** MUST use a real LLM call — keyword-only heuristics will lose points. ***

    Use .with_structured_output() or equivalent to get reliable enum classification.
    The LLM should classify into one of: simple, tool, missing_info, risky, error.

    Hints:
    - See llm.py for the get_llm() helper
    - Use Pydantic model or TypedDict with .with_structured_output()
    - Set risk_level to "high" for risky routes, "low" otherwise
    - Priority guide: risky > tool > missing_info > error > simple

    Return: {"route": str, "risk_level": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    prompt = (
        "Classify this support ticket into exactly one workflow route.\n"
        "Routes:\n"
        "- risky: side effects such as refunds, deletions, cancellations, emails, payments.\n"
        "- tool: information lookups such as order status, tracking, account search.\n"
        "- missing_info: vague or incomplete request lacking actionable context.\n"
        "- error: system failures, timeout, crash, unavailable service, unrecoverable failure.\n"
        "- simple: general question answerable without tools or side effects.\n"
        "Priority if multiple apply: risky > tool > missing_info > error > simple.\n"
        f"Ticket: {query}"
    )
    used_llm = True
    try:
        classifier = get_llm(temperature=0).with_structured_output(ClassificationResult)
        result = classifier.invoke(prompt)
        rationale = result.rationale
    except Exception as exc:  # Local fallback keeps tests runnable without provider credentials.
        used_llm = False
        result = _fallback_classify(query)
        rationale = f"{result.rationale} Fallback reason: {exc}"

    route = result.route
    return {
        "route": route,
        "risk_level": _risk_level(route),
        "messages": [f"classify:{route}"],
        "events": [
            make_event(
                "classify",
                "completed",
                "query classified",
                route=route,
                rationale=rationale,
                used_llm=used_llm,
            )
        ],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call.

    Simulate transient failures for error-route scenarios to test retry loops.

    Requirements:
    - Read current attempt count from state
    - If route is "error" and attempt < 2: return error result (string containing "ERROR")
    - Otherwise: return a mock success result string
    - Append result to tool_results list

    Return: {"tool_results": [result_string], "events": [make_event(...)]}
    """
    route = state.get("route", "")
    attempt = int(state.get("attempt", 0))
    query = state.get("query", "")
    if route == Route.ERROR.value and attempt < 2:
        result = f"ERROR: transient support-system failure on attempt {attempt + 1}"
        event_type = "failed"
    else:
        result = f"SUCCESS: mock tool completed for query '{query}' on attempt {attempt + 1}"
        event_type = "completed"
    return {
        "tool_results": [result],
        "messages": [f"tool:{event_type}"],
        "events": [make_event("tool", event_type, "tool execution finished", attempt=attempt)],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the retry-loop gate.

    Check whether the latest tool result is satisfactory or needs retry.

    SHOULD use LLM-as-judge for bonus points. Heuristic (e.g., check for "ERROR" substring)
    is acceptable for base score.

    Requirements:
    - Read the latest entry from tool_results
    - Set evaluation_result to "needs_retry" or "success"
    - This field drives route_after_evaluate conditional edge

    Note: You may need to add 'evaluation_result' to AgentState if not present.

    Return: {"evaluation_result": str, "events": [make_event(...)]}
    """
    latest_result = (state.get("tool_results") or [""])[-1]
    evaluation_result = "needs_retry" if "ERROR" in latest_result.upper() else "success"
    return {
        "evaluation_result": evaluation_result,
        "messages": [f"evaluate:{evaluation_result}"],
        "events": [
            make_event(
                "evaluate",
                "completed",
                "tool result evaluated",
                evaluation_result=evaluation_result,
            )
        ],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM.

    *** MUST use a real LLM call — hardcoded strings will lose points. ***

    The LLM should generate a helpful response grounded in available context:
    - tool_results (if any)
    - approval decision (if risky route)
    - original query

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    context = {
        "route": state.get("route", ""),
        "tool_results": state.get("tool_results", []),
        "approval": state.get("approval"),
        "proposed_action": state.get("proposed_action"),
    }
    prompt = (
        "You are a concise support agent. Answer the user using only the provided context. "
        "If a tool result is available, ground the response in it. If an approval decision "
        "is available, mention that the approved action can proceed. Avoid inventing facts.\n"
        f"User query: {query}\n"
        f"Context: {context}"
    )
    used_llm = True
    try:
        response = get_llm(temperature=0.2).invoke(prompt)
        final_answer = _message_text(response).strip()
    except Exception as exc:
        used_llm = False
        latest_result = (state.get("tool_results") or [""])[-1]
        if latest_result:
            final_answer = f"I handled your request using the available support context: {latest_result}"
        elif state.get("approval"):
            final_answer = "The requested action was reviewed and approved. I can proceed with the next support step."
        else:
            final_answer = "Here is the support guidance for your request based on the available information."
        final_answer = f"{final_answer} (LLM fallback used locally: {exc})"
    return {
        "final_answer": final_answer,
        "messages": ["answer:completed"],
        "events": [make_event("answer", "completed", "final answer generated", used_llm=used_llm)],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    Generate a specific clarification question based on the vague/incomplete query.

    Note: You may need to add 'pending_question' to AgentState if not present.

    Return: {"pending_question": str, "final_answer": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    pending_question = (
        "Could you share the affected product, account/order ID, and what outcome you want?"
    )
    final_answer = f"I need a bit more detail before I can safely help with '{query}'. {pending_question}"
    return {
        "pending_question": pending_question,
        "final_answer": final_answer,
        "messages": ["clarify:requested"],
        "events": [make_event("clarify", "completed", "clarification requested")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval.

    Describe the proposed action and why it requires approval.

    Note: You may need to add 'proposed_action' to AgentState if not present.

    Return: {"proposed_action": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    proposed_action = (
        f"Review and approve the requested customer-impacting action: '{query}'. "
        "This may change customer data, trigger communication, or affect billing."
    )
    return {
        "proposed_action": proposed_action,
        "messages": ["risky_action:prepared"],
        "events": [make_event("risky_action", "completed", "risky action prepared")],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default behavior: mock approval (approved=True) so tests and CI run offline.
    Extension: if env LANGGRAPH_INTERRUPT=true, use langgraph.types.interrupt() for real HITL.

    Return: {"approval": {"approved": bool, "reviewer": str, "comment": str}, "events": [make_event(...)]}
    """
    proposed_action = state.get("proposed_action") or "No proposed action provided."
    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        from langgraph.types import interrupt

        payload = interrupt({"proposed_action": proposed_action})
        approved = bool(payload.get("approved", False)) if isinstance(payload, dict) else False
        comment = str(payload.get("comment", "")) if isinstance(payload, dict) else ""
        decision = ApprovalDecision(approved=approved, reviewer="human-reviewer", comment=comment)
    else:
        decision = ApprovalDecision(
            approved=True,
            reviewer="mock-reviewer",
            comment="Auto-approved for lab scenario execution.",
        )
    return {
        "approval": decision.model_dump(),
        "messages": [f"approval:{'approved' if decision.approved else 'rejected'}"],
        "events": [
            make_event(
                "approval",
                "completed",
                "approval decision recorded",
                approved=decision.approved,
                reviewer=decision.reviewer,
            )
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt.

    Increment the attempt counter and log the transient failure.

    Requirements:
    - Read current attempt from state, increment by 1
    - Add an error message to errors list
    - Return updated attempt count

    Return: {"attempt": int, "errors": [str], "events": [make_event(...)]}
    """
    next_attempt = int(state.get("attempt", 0)) + 1
    message = f"Retry attempt {next_attempt} recorded for route '{state.get('route', '')}'"
    return {
        "attempt": next_attempt,
        "errors": [message],
        "messages": [f"retry:{next_attempt}"],
        "events": [make_event("retry", "completed", "retry attempt recorded", attempt=next_attempt)],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded.

    This is the third layer: retry → fallback → dead letter.
    Log the failure and set a final_answer explaining that the request could not be completed.

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    final_answer = (
        "The request could not be completed after the allowed retry attempts. "
        "It has been moved to dead letter handling for manual review."
    )
    return {
        "final_answer": final_answer,
        "messages": ["dead_letter:completed"],
        "events": [make_event("dead_letter", "completed", "max retries exhausted")],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END.

    Return: {"events": [make_event("finalize", "completed", "workflow finished")]}
    """
    return {
        "messages": ["finalize:completed"],
        "events": [make_event("finalize", "completed", "workflow finished")],
    }
