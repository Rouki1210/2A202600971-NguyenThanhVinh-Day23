from langgraph_agent_lab.metrics import metric_from_state, summarize_metrics
from langgraph_agent_lab.state import make_event


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _state(
    route="simple",
    final_answer="ok",
    events=None,
    errors=None,
    approval=None,
    scenario_id="S",
):
    return {
        "scenario_id": scenario_id,
        "route": route,
        "final_answer": final_answer,
        "events": events if events is not None else [],
        "errors": errors if errors is not None else [],
        "approval": approval,
    }


# ──────────────────────────────────────────────
# metric_from_state — success / failure logic
# ──────────────────────────────────────────────

def test_no_final_answer_is_failure():
    """Missing final_answer should mark the metric as failed."""
    metric = metric_from_state(_state(final_answer=None), "simple", False)
    assert metric.success is False


def test_errors_present_is_failure():
    """Any error in the state should mark the metric as failed."""
    metric = metric_from_state(
        _state(errors=["something went wrong"]), "simple", False
    )
    assert metric.success is False


def test_multiple_errors_all_captured():
    """All errors in the state should be captured in the metric."""
    errors = ["err1", "err2", "err3"]
    metric = metric_from_state(_state(errors=errors), "simple", False)
    assert metric.success is False
    assert len(metric.errors) == 3


def test_approval_required_but_missing_is_failure():
    """approval_required=True but approval=None → failure."""
    metric = metric_from_state(_state(approval=None), "simple", approval_required=True)
    assert metric.success is False


def test_approval_required_and_granted_is_success():
    """approval_required=True and approval=True → success (route also matches)."""
    metric = metric_from_state(_state(approval=True), "simple", approval_required=True)
    assert metric.success is True


def test_approval_not_required_and_present_does_not_break():
    """approval_required=False but approval exists — should still succeed."""
    metric = metric_from_state(_state(approval=True), "simple", approval_required=False)
    assert metric.success is True


# ──────────────────────────────────────────────
# metric_from_state — nodes_visited
# ──────────────────────────────────────────────

def test_nodes_visited_empty_events():
    """Zero events → nodes_visited == 0."""
    metric = metric_from_state(_state(events=[]), "simple", False)
    assert metric.nodes_visited == 0


def test_nodes_visited_single_event():
    events = [make_event("intake", "completed", "ok")]
    metric = metric_from_state(_state(events=events), "simple", False)
    assert metric.nodes_visited == 1


def test_nodes_visited_many_events():
    events = [
        make_event("intake", "completed", "ok"),
        make_event("tool_call", "completed", "result"),
        make_event("answer", "completed", "ok"),
        make_event("review", "completed", "approved"),
    ]
    metric = metric_from_state(_state(events=events), "simple", False)
    assert metric.nodes_visited == 4


# ──────────────────────────────────────────────
# metric_from_state — route matching
# ──────────────────────────────────────────────

def test_route_mismatch_simple_vs_tool():
    metric = metric_from_state(_state(route="tool"), "simple", False)
    assert metric.success is False


def test_route_mismatch_tool_vs_simple():
    metric = metric_from_state(_state(route="simple"), "tool", False)
    assert metric.success is False


def test_route_match_tool():
    metric = metric_from_state(_state(route="tool"), "tool", False)
    assert metric.success is True


def test_route_match_stored_in_metric():
    """The metric should carry the actual route from state."""
    metric = metric_from_state(_state(route="tool"), "simple", False)
    assert metric.route == "tool"


def test_scenario_id_stored_in_metric():
    metric = metric_from_state(_state(scenario_id="scenario-42"), "simple", False)
    assert metric.scenario_id == "scenario-42"


# ──────────────────────────────────────────────
# summarize_metrics — aggregation
# ──────────────────────────────────────────────

def _make_metric(route="simple", expected_route="simple", final_answer="ok",
                 errors=None, scenario_id="S", approval=None, approval_required=False):
    return metric_from_state(
        _state(route=route, final_answer=final_answer,
               errors=errors or [], approval=approval, scenario_id=scenario_id),
        expected_route=expected_route,
        approval_required=approval_required,
    )


def test_summarize_all_success():
    metrics = [_make_metric(scenario_id=str(i)) for i in range(5)]
    report = summarize_metrics(metrics)
    assert report.total_scenarios == 5
    assert report.success_rate == 1.0


def test_summarize_all_failure():
    metrics = [_make_metric(final_answer=None, scenario_id=str(i)) for i in range(3)]
    report = summarize_metrics(metrics)
    assert report.total_scenarios == 3
    assert report.success_rate == 0.0


def test_summarize_partial_success_rate():
    """3 success + 1 failure → success_rate == 0.75."""
    metrics = [
        _make_metric(scenario_id="1"),
        _make_metric(scenario_id="2"),
        _make_metric(scenario_id="3"),
        _make_metric(scenario_id="4", final_answer=None),
    ]
    report = summarize_metrics(metrics)
    assert report.total_scenarios == 4
    assert abs(report.success_rate - 0.75) < 1e-9


def test_summarize_single_success():
    report = summarize_metrics([_make_metric()])
    assert report.total_scenarios == 1
    assert report.success_rate == 1.0


def test_summarize_single_failure():
    report = summarize_metrics([_make_metric(final_answer=None)])
    assert report.total_scenarios == 1
    assert report.success_rate == 0.0


def test_summarize_empty_list():
    """Edge case: no metrics — should not crash and total_scenarios == 0."""
    report = summarize_metrics([])
    assert report.total_scenarios == 0


def test_summarize_success_rate_bounds():
    """success_rate must always be in [0.0, 1.0]."""
    metrics = [_make_metric(scenario_id=str(i)) for i in range(10)]
    report = summarize_metrics(metrics)
    assert 0.0 <= report.success_rate <= 1.0


def test_summarize_mixed_routes():
    """Route mismatches count as failures in the summary."""
    metrics = [
        _make_metric(route="simple", expected_route="simple", scenario_id="1"),  # pass
        _make_metric(route="tool",   expected_route="simple", scenario_id="2"),  # fail
    ]
    report = summarize_metrics(metrics)
    assert report.total_scenarios == 2
    assert report.success_rate == 0.5