from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ActionKey = tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class WorkflowCandidate:
    source: str
    target: str
    relation: str
    rank: int
    description: str = ""

    @property
    def action_key(self) -> ActionKey:
        return (self.source, self.target, self.relation)


@dataclass(frozen=True, slots=True)
class WorkflowPredictionView:
    query_text: str
    candidates: tuple[WorkflowCandidate, ...]
    node_profile_by_id: tuple[tuple[str, str], ...] = ()

    def node_profile(self, node_id: str) -> str:
        for current_node_id, text in self.node_profile_by_id:
            if current_node_id == node_id:
                return text
        return ""


@dataclass(slots=True)
class WorkflowPolicyResult:
    dataset_name: str
    total_steps: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    hit_counts: dict[str, int]
    relation_hit_at_1: dict[str, float]
    eval_files: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "total_steps": self.total_steps,
            "hit_at_1": self.hit_at_1,
            "hit_at_3": self.hit_at_3,
            "hit_at_5": self.hit_at_5,
            "hit_counts": self.hit_counts,
            "relation_hit_at_1": self.relation_hit_at_1,
            "eval_files": self.eval_files,
        }


def prediction_view_from_payload(payload: dict[str, Any]) -> WorkflowPredictionView:
    """Return only the fields that are visible at prediction time."""
    prediction = payload.get("prediction") or {}
    source = prediction.get("source")
    query_text = str(prediction.get("query") or "")
    candidates: list[WorkflowCandidate] = []
    seen: set[ActionKey] = set()
    if source is not None:
        source_text = str(source)
        for rank, candidate in enumerate(prediction.get("transition_candidates") or []):
            relation = str(candidate.get("relation") or "").strip().lower()
            if not relation:
                continue
            description = str(candidate.get("description") or "")
            for target in candidate.get("targets") or []:
                key = (source_text, str(target), relation)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    WorkflowCandidate(
                        source=key[0],
                        target=key[1],
                        relation=key[2],
                        rank=rank,
                        description=description,
                    )
                )

    node_profile_by_id: list[tuple[str, str]] = []
    for node_id, node in sorted(((payload.get("graph") or {}).get("nodes") or {}).items()):
        node_profile_by_id.append((str(node_id), str(node.get("profile") or "")))
    return WorkflowPredictionView(
        query_text=query_text,
        candidates=tuple(candidates),
        node_profile_by_id=tuple(node_profile_by_id),
    )


def observed_action_from_payload(payload: dict[str, Any]) -> ActionKey | None:
    prediction = payload.get("prediction") or {}
    label = prediction.get("label") or {}
    source = prediction.get("source")
    relation = str(label.get("relation") or "").strip().lower()
    targets = [str(target) for target in (label.get("targets") or []) if str(target)]
    if source is None or not relation or not targets:
        return None
    return (str(source), targets[0], relation)


def rank_workflow_candidates(
    view: WorkflowPredictionView,
    *,
    previous_observed_action: ActionKey | None,
    scenario: str,
) -> list[ActionKey]:
    ranked: list[ActionKey] = []

    def add(action: ActionKey | None) -> None:
        if action is not None and action not in ranked:
            ranked.append(action)

    if previous_observed_action and previous_observed_action[2] == "delegate":
        add(
            _first_candidate(
                view.candidates,
                relation="delegate_return",
                target=previous_observed_action[0],
            )
        )
    if previous_observed_action and previous_observed_action[2] == "delegate_return":
        add(_first_candidate(view.candidates, relation="activate"))

    if scenario == "research":
        add(_first_candidate(view.candidates, relation="delegate", exclude_self=True))

    add(_first_candidate(view.candidates, relation="activate"))
    add(_first_candidate(view.candidates, relation="retry"))
    add(_first_candidate(view.candidates, relation="delegate_return"))
    add(_first_candidate(view.candidates, relation="delegate"))

    for candidate in view.candidates:
        add(candidate.action_key)
    return ranked


def evaluate_workflow_policy_payloads(
    payloads_by_file: list[tuple[str, list[dict[str, Any]]]],
    *,
    dataset_name: str,
    scenario: str,
) -> WorkflowPolicyResult:
    hit_counts = Counter({1: 0, 3: 0, 5: 0})
    relation_hits: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    total = 0
    eval_files: list[str] = []
    for file_name, payloads in payloads_by_file:
        eval_files.append(file_name)
        previous_observed_action: ActionKey | None = None
        for payload in payloads:
            view = prediction_view_from_payload(payload)
            ranked = rank_workflow_candidates(
                view,
                previous_observed_action=previous_observed_action,
                scenario=scenario,
            )
            expected = observed_action_from_payload(payload)
            if expected is None:
                continue
            total += 1
            relation_hits[expected[2]][1] += 1
            for hit_k in (1, 3, 5):
                if expected in ranked[:hit_k]:
                    hit_counts[hit_k] += 1
            if expected in ranked[:1]:
                relation_hits[expected[2]][0] += 1
            previous_observed_action = expected

    return WorkflowPolicyResult(
        dataset_name=dataset_name,
        total_steps=total,
        hit_at_1=hit_counts[1] / total if total else 0.0,
        hit_at_3=hit_counts[3] / total if total else 0.0,
        hit_at_5=hit_counts[5] / total if total else 0.0,
        hit_counts={str(key): int(value) for key, value in hit_counts.items()},
        relation_hit_at_1={
            relation: (counts[0] / counts[1] if counts[1] else 0.0)
            for relation, counts in sorted(relation_hits.items())
        },
        eval_files=eval_files,
    )


def split_acg_nap_paths(
    root: Path,
    scenario: str,
    *,
    max_files_per_dataset: int,
    train_fraction: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    paths = sorted((root / scenario).glob("*.jsonl"))
    if max_files_per_dataset > 0:
        paths = paths[:max_files_per_dataset]
    if not paths:
        raise FileNotFoundError(f"No ACG-NAP JSONL files found for {scenario} under {root}")
    random.Random(seed).shuffle(paths)
    if len(paths) == 1:
        return paths, paths
    train_count = max(1, min(len(paths) - 1, math.ceil(len(paths) * train_fraction)))
    return paths[:train_count], paths[train_count:]


def evaluate_acg_nap_workflow_policy(
    root: Path,
    *,
    scenarios: tuple[str, ...] = ("coding", "research"),
    max_files_per_dataset: int = 15,
    train_fraction: float = 0.8,
    seed: int = 7,
) -> list[WorkflowPolicyResult]:
    results = [
        _evaluate_dataset(
            root,
            scenario,
            max_files_per_dataset=max_files_per_dataset,
            train_fraction=train_fraction,
            seed=seed,
        )
        for scenario in scenarios
    ]
    if len(results) <= 1:
        return results
    total_steps = sum(item.total_steps for item in results)
    combined = WorkflowPolicyResult(
        dataset_name="combined",
        total_steps=total_steps,
        hit_at_1=_weighted_average(results, "hit_at_1", total_steps),
        hit_at_3=_weighted_average(results, "hit_at_3", total_steps),
        hit_at_5=_weighted_average(results, "hit_at_5", total_steps),
        hit_counts={
            str(hit_k): sum(item.hit_counts[str(hit_k)] for item in results)
            for hit_k in (1, 3, 5)
        },
        relation_hit_at_1={},
        eval_files=[],
    )
    return [combined, *results]


def _evaluate_dataset(
    root: Path,
    scenario: str,
    *,
    max_files_per_dataset: int,
    train_fraction: float,
    seed: int,
) -> WorkflowPolicyResult:
    _, eval_paths = split_acg_nap_paths(
        root,
        scenario,
        max_files_per_dataset=max_files_per_dataset,
        train_fraction=train_fraction,
        seed=seed,
    )
    payloads_by_file = [
        (path.name, _load_jsonl(path))
        for path in eval_paths
    ]
    return evaluate_workflow_policy_payloads(
        payloads_by_file,
        dataset_name=scenario,
        scenario=scenario,
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _first_candidate(
    candidates: tuple[WorkflowCandidate, ...],
    *,
    relation: str | None = None,
    target: str | None = None,
    exclude_self: bool = False,
) -> ActionKey | None:
    for candidate in candidates:
        if relation is not None and candidate.relation != relation:
            continue
        if target is not None and candidate.target != target:
            continue
        if exclude_self and candidate.source == candidate.target:
            continue
        return candidate.action_key
    return None


def _weighted_average(
    results: list[WorkflowPolicyResult],
    field_name: str,
    total_steps: int,
) -> float:
    if total_steps <= 0:
        return 0.0
    return sum(getattr(item, field_name) * item.total_steps for item in results) / total_steps
