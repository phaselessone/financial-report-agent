from src.agent.dependency_reasoning import DependencyEdge, SubQuestionResult, assess_coverage, normalize_dependency_edges
from src.agent.nodes.dependency_gate import dependency_gate, make_dependency_gate

def test_complete_coverage_allows_deterministic_conclusion():
    report = assess_coverage(["q1", "q2"], [{"id": "q1", "status": "completed", "evidence_ids": ["e1"]}, {"id": "q2", "status": "completed", "evidence_ids": ["e2"]}])
    assert report["coverage_ratio"] == 1.0 and report["deterministic_conclusion_allowed"]

def test_partial_coverage_disallows_deterministic_conclusion():
    report = assess_coverage(["q1", "q2"], [{"id": "q1", "status": "completed", "evidence_ids": ["e1"]}])
    assert report["missing"] == ["q2"] and report["partial"] and not report["deterministic_conclusion_allowed"]

def test_no_evidence_abstains_even_without_required_ids():
    report = assess_coverage([], [{"id": "q1", "status": "pending"}])
    assert report["abstain"]

def test_completed_without_own_evidence_is_not_satisfied_by_global_pool():
    report = assess_coverage(
        ["q1"],
        [{"id": "q1", "status": "completed"}],
        evidence_pool={"unrelated": {"text": "different sub-question"}},
    )
    assert report["missing"] == ["q1"]
    assert report["decision"] == "abstain"

def test_dependency_blocks_target_until_source_complete():
    report = assess_coverage(["q2"], [{"id": "q2", "status": "completed", "evidence_ids": ["e2"]}], edges=[DependencyEdge("q1", "q2")])
    assert report["missing"] == ["q2"] and report["blocked_by"] == {"q2": ["q1"]}

def test_transitive_dependencies_propagate_missing_source():
    report = assess_coverage(["q2", "q3"], [{"id": "q2", "status": "completed", "evidence_ids": ["e2"]}, {"id": "q3", "status": "completed", "evidence_ids": ["e3"]}], edges=[DependencyEdge("q2", "q3"), DependencyEdge("q1", "q2")])
    assert report["missing"] == ["q2", "q3"]

def test_edges_can_be_declared_on_subquestion_payload():
    edges = normalize_dependency_edges([], [{"id": "q2", "depends_on": ["q1"]}])
    assert edges[0].source == "q1" and edges[0].target == "q2"

def test_gate_writes_observable_partial_fields():
    state = {"sub_questions": [{"id": "q1"}, {"id": "q2"}], "sub_question_results": [{"id": "q1", "status": "completed", "evidence_ids": ["e1"]}], "evidence_pool": {"e1": {}}}
    out = dependency_gate(state)
    assert out["coverage_report"]["coverage_ratio"] == 0.5 and out["partial_answer"] and not out["deterministic_conclusion_allowed"]
    assert out["missing_information"] == "q2"

def test_dataclass_roundtrip_and_factory():
    item = SubQuestionResult(id="q1", status="completed", evidence_ids=("e1",))
    assert SubQuestionResult.from_mapping(item).to_dict()["completed"] and make_dependency_gate() is dependency_gate
