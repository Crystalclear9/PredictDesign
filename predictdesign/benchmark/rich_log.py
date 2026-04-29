from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..messages import Message, MessageAction
from ..prediction import GraphActionType, PredictedGraphAction
from ..temporal_graph import TemporalEdge, TemporalGraph, TemporalNode
from .types import BenchmarkEpisode, EpisodeStep


INFO_COMBINATIONS: tuple[tuple[str, ...], ...] = (
    ("node_outputs",),
    ("graph_structure",),
    ("query",),
    ("node_outputs", "graph_structure"),
    ("node_outputs", "query"),
    ("graph_structure", "query"),
    ("node_outputs", "graph_structure", "query"),
)


@dataclass(slots=True)
class RichLogExportResult:
    path: str
    record_count: int
    episode_count: int
    dataset_counts: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class MLPCombinationResult:
    dataset_name: str
    combination: str
    signals: list[str]
    sample_count: int
    train_count: int
    val_count: int
    class_count: int
    input_dim: int
    hidden_dims: list[int]
    batch_size: int
    val_accuracy: float
    val_accuracy_std: float
    run_count: int
    best_epoch: int
    signal_dims: list[int]
    signal_projection_dim: int


@dataclass(slots=True)
class MLPDatasetResult:
    dataset_name: str
    sample_count: int
    train_count: int
    val_count: int
    class_labels: list[str]
    max_graph_nodes: int
    graph_vector_dim: int
    text_vector_dim: int
    combinations: list[MLPCombinationResult]


@dataclass(slots=True)
class MLPTrainingResult:
    log_path: str
    output_dir: str
    report_path: str
    csv_path: str
    chart_path: str
    device: str
    cuda_available: bool
    sample_count: int
    train_count: int
    val_count: int
    label_mode: str
    sentence_transformer_model: str
    sentence_transformer_dim: int
    repeat_count: int
    datasets: list[MLPDatasetResult]
    combinations: list[MLPCombinationResult]


class _SignalTower(nn.Module):
    def __init__(self, input_dim: int, projection_dim: int) -> None:
        super().__init__()
        hidden_dim = max(64, projection_dim // 2)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(projection_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class _FusionMLP(nn.Module):
    def __init__(
        self,
        *,
        signal_dims: tuple[int, ...],
        projection_dim: int,
        hidden_dims: tuple[int, ...],
        class_count: int,
    ) -> None:
        super().__init__()
        self.signal_dims = signal_dims
        self.towers = nn.ModuleList(
            _SignalTower(input_dim=signal_dim, projection_dim=projection_dim)
            for signal_dim in signal_dims
        )
        layers: list[nn.Module] = []
        previous_dim = projection_dim * len(signal_dims)
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(p=0.14),
                ]
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, max(class_count, 1)))
        self.head = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        parts = torch.split(features, self.signal_dims, dim=-1)
        encoded = [tower(part) for tower, part in zip(self.towers, parts)]
        fused = torch.cat(encoded, dim=-1)
        return self.head(fused)


def write_rich_log(
    output_path: str | Path,
    episodes: Iterable[BenchmarkEpisode],
    *,
    context_dim: int,
    device: str = "cpu",
) -> RichLogExportResult:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    episode_list = list(episodes)
    dataset_counts: dict[str, int] = {}
    record_count = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in iter_rich_log_records(
            episode_list,
            context_dim=context_dim,
            device=device,
        ):
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            dataset_name = str(record.get("dataset_name", "unknown"))
            dataset_counts[dataset_name] = dataset_counts.get(dataset_name, 0) + 1
            record_count += 1
    return RichLogExportResult(
        path=str(output_path),
        record_count=record_count,
        episode_count=len(episode_list),
        dataset_counts=dataset_counts,
    )


def iter_rich_log_records(
    episodes: Iterable[BenchmarkEpisode],
    *,
    context_dim: int,
    device: str = "cpu",
) -> Iterable[dict[str, Any]]:
    sample_index = 0
    for episode in episodes:
        graph = _build_initial_graph(episode, context_dim=context_dim, device=device)
        current_query = ""
        initial_snapshot = _graph_snapshot(graph, time_value=0.0)
        for step_index, step in enumerate(episode.steps):
            graph_before = _graph_snapshot(graph, time_value=step.observation_time)
            step_query = _query_text_from_messages(step.messages)
            if step_query:
                current_query = step_query
            _apply_context_updates(graph, step)
            _apply_actions_to_graph(graph, step.observed_actions, context_dim=context_dim, device=device)
            graph_after = _graph_snapshot(graph, time_value=step.observation_time)
            target_actions = step.supervision_actions
            record = {
                "schema_version": 1,
                "sample_id": f"{episode.dataset_name}:{episode.episode_id}:{step_index}",
                "sample_index": sample_index,
                "dataset_name": episode.dataset_name,
                "episode_id": episode.episode_id,
                "step_index": step_index,
                "observation_time": float(step.observation_time),
                "query": current_query,
                "nodes": graph_after["nodes"],
                "graph_structure": graph_after,
                "graph_before_step": graph_before,
                "initial_graph": initial_snapshot,
                "messages": [_message_to_dict(message) for message in step.messages],
                "observed_actions": [_action_to_dict(action) for action in step.observed_actions],
                "target_actions": [_action_to_dict(action) for action in target_actions],
                "target_action_type": target_actions[0].action_type.value if target_actions else "no_op",
                "target_action_signature": _action_signature(target_actions[0]) if target_actions else "no_op",
                "target_action_count": len(target_actions),
            }
            yield record
            sample_index += 1


def train_mlp_on_rich_log(
    log_path: str | Path,
    output_dir: str | Path,
    *,
    max_samples: int = 100,
    train_fraction: float = 0.8,
    feature_dim: int = 256,
    hidden_dim: int | None = None,
    epochs: int = 60,
    learning_rate: float = 1e-3,
    seed: int = 7,
    label_mode: str = "action_type",
    device: str | None = "auto",
    sentence_transformer_model: str = "all-MiniLM-L6-v2",
    sentence_transformer_batch_size: int = 64,
    hidden_dims: tuple[int, ...] | None = None,
    repeat_count: int = 5,
) -> MLPTrainingResult:
    if label_mode not in {"action_type", "action_signature"}:
        raise ValueError("label_mode must be 'action_type' or 'action_signature'.")
    records = _load_rich_log_records(log_path)
    dataset_records = _group_records_by_dataset(records, max_samples=max_samples, seed=seed)
    sample_count = sum(len(items) for items in dataset_records.values())
    if sample_count < 2:
        raise ValueError("Need at least 2 rich-log records to train and validate an MLP.")
    run_device = _resolve_training_device(device)
    _configure_training_device(run_device)
    text_cache, text_dim, resolved_model_name = _build_text_embedding_cache(
        dataset_records,
        sentence_transformer_model=sentence_transformer_model,
        sentence_transformer_batch_size=sentence_transformer_batch_size,
        device=run_device,
        fallback_dim=feature_dim,
    )

    dataset_results: list[MLPDatasetResult] = []
    overall_results: list[MLPCombinationResult] = []
    total_train_count = 0
    total_val_count = 0
    for dataset_index, dataset_name in enumerate(_ordered_dataset_names(dataset_records)):
        samples = dataset_records[dataset_name]
        if len(samples) < 2:
            continue
        labels = [_label_for_record(record, label_mode=label_mode) for record in samples]
        class_labels = sorted(set(labels))
        label_to_index = {label: index for index, label in enumerate(class_labels)}
        y = torch.tensor([label_to_index[label] for label in labels], dtype=torch.long)
        train_indices, val_indices = _split_train_val_indices(
            labels=y.tolist(),
            train_fraction=train_fraction,
            seed=seed + dataset_index,
        )
        max_graph_nodes = _max_graph_nodes(samples)
        combinations: list[MLPCombinationResult] = []
        for signals in INFO_COMBINATIONS:
            features, signal_dims = _build_feature_matrix(
                samples,
                signals=signals,
                text_cache=text_cache,
                text_dim=text_dim,
                max_graph_nodes=max_graph_nodes,
            )
            resolved_hidden_dims = _resolve_hidden_dims(
                input_dim=int(features.size(1)),
                sample_count=len(samples),
                hidden_dim=hidden_dim,
                explicit_hidden_dims=hidden_dims,
            )
            signal_projection_dim = _resolve_signal_projection_dim(
                signal_dims=signal_dims,
                sample_count=len(samples),
            )
            batch_size = _resolve_batch_size(
                train_count=int(train_indices.numel()),
                device=run_device,
            )
            combination_result = _train_one_combination(
                dataset_name=dataset_name,
                features=features,
                labels=y,
                train_indices=train_indices,
                val_indices=val_indices,
                signals=signals,
                class_count=len(class_labels),
                signal_dims=signal_dims,
                signal_projection_dim=signal_projection_dim,
                hidden_dims=resolved_hidden_dims,
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=learning_rate,
                seed=seed + dataset_index,
                device=run_device,
                repeat_count=repeat_count,
            )
            combinations.append(combination_result)
        dataset_results.append(
            MLPDatasetResult(
                dataset_name=dataset_name,
                sample_count=len(samples),
                train_count=int(train_indices.numel()),
                val_count=int(val_indices.numel()),
                class_labels=class_labels,
                max_graph_nodes=max_graph_nodes,
                graph_vector_dim=max_graph_nodes * max_graph_nodes,
                text_vector_dim=text_dim,
                combinations=combinations,
            )
        )
        total_train_count += int(train_indices.numel())
        total_val_count += int(val_indices.numel())

    for signals in INFO_COMBINATIONS:
        combination_name = "+".join(signals)
        matching = [
            result
            for dataset_result in dataset_results
            for result in dataset_result.combinations
            if result.combination == combination_name
        ]
        if not matching:
            continue
        overall_results.append(
            MLPCombinationResult(
                dataset_name="overall_mean",
                combination=combination_name,
                signals=list(signals),
                sample_count=sum(item.sample_count for item in matching),
                train_count=sum(item.train_count for item in matching),
                val_count=sum(item.val_count for item in matching),
                class_count=max(item.class_count for item in matching),
                input_dim=0,
                hidden_dims=[],
                batch_size=0,
                val_accuracy=float(sum(item.val_accuracy for item in matching) / len(matching)),
                val_accuracy_std=float(torch.tensor([item.val_accuracy for item in matching]).std(unbiased=False).item()),
                run_count=max(item.run_count for item in matching),
                best_epoch=int(round(sum(item.best_epoch for item in matching) / len(matching))),
                signal_dims=[],
                signal_projection_dim=0,
            )
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "rich_log_mlp_results.json"
    csv_path = output_dir / "rich_log_mlp_accuracy.csv"
    chart_path = output_dir / "rich_log_mlp_accuracy.png"
    result = MLPTrainingResult(
        log_path=str(Path(log_path)),
        output_dir=str(output_dir),
        report_path=str(report_path),
        csv_path=str(csv_path),
        chart_path=str(chart_path),
        device=str(run_device),
        cuda_available=torch.cuda.is_available(),
        sample_count=sample_count,
        train_count=total_train_count,
        val_count=total_val_count,
        label_mode=label_mode,
        sentence_transformer_model=resolved_model_name,
        sentence_transformer_dim=text_dim,
        repeat_count=int(max(repeat_count, 1)),
        datasets=dataset_results,
        combinations=overall_results,
    )
    _write_mlp_report(report_path, result)
    _write_mlp_csv(csv_path, dataset_results)
    _write_accuracy_png(chart_path, dataset_results)
    return result


def _resolve_training_device(device: str | None) -> torch.device:
    requested = (device or "auto").strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for MLP training, but torch.cuda.is_available() is False.")
    return resolved


def _configure_training_device(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def _build_initial_graph(
    episode: BenchmarkEpisode,
    *,
    context_dim: int,
    device: str,
) -> TemporalGraph:
    graph = TemporalGraph(context_dim=context_dim, device=device)
    for node in episode.initial_nodes:
        graph.add_node(node)
    for edge in episode.initial_edges:
        if edge.source_node_id in graph.nodes and edge.target_node_id in graph.nodes:
            graph.add_edge(
                TemporalEdge(
                    source_node_id=edge.source_node_id,
                    target_node_id=edge.target_node_id,
                    start_time=edge.start_time,
                    end_time=edge.end_time,
                )
            )
    for source_node_id, target_node_id in episode.initial_structural_edges:
        if source_node_id in graph.nodes and target_node_id in graph.nodes:
            graph.add_structural_edge(source_node_id, target_node_id)
    return graph


def _graph_snapshot(graph: TemporalGraph, time_value: float) -> dict[str, Any]:
    node_order = sorted(graph.nodes)
    adjacency = graph.adjacency_matrix(
        time_value=time_value,
        node_order=node_order,
        include_structural=True,
    )
    active_edges = graph.active_edges(time_value)
    nodes = [_node_to_dict(graph.nodes[node_id]) for node_id in node_order]
    return {
        "time": float(time_value),
        "node_order": node_order,
        "node_count": len(node_order),
        "active_edge_count": len(active_edges),
        "all_edge_count": len(graph.edges),
        "structural_edge_count": len(graph.structural_edges),
        "nodes": nodes,
        "active_edges": [_edge_to_dict(edge) for edge in active_edges],
        "all_edges": [_edge_to_dict(edge) for edge in graph.edges],
        "structural_edges": [
            {"source_node_id": source_node_id, "target_node_id": target_node_id}
            for source_node_id, target_node_id in sorted(graph.structural_edges)
        ],
        "adjacency": adjacency.detach().cpu().int().tolist(),
    }


def _node_to_dict(node: TemporalNode) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "role": node.role,
        "output_text": str(node.context_text or ""),
        "context_vector": _tensor_to_list(node.context),
    }


def _edge_to_dict(edge: TemporalEdge) -> dict[str, Any]:
    return {
        "source_node_id": edge.source_node_id,
        "target_node_id": edge.target_node_id,
        "start_time": float(edge.start_time),
        "end_time": float(edge.end_time),
    }


def _message_to_dict(message: Message) -> dict[str, Any]:
    return {
        "source_node_id": message.source_node_id,
        "target_node_id": message.target_node_id,
        "time": float(message.time),
        "action": message.action.value,
        "raw_text": str(message.metadata.get("raw_text", "")) if message.metadata else "",
        "query_text": str(message.metadata.get("query_text", "")) if message.metadata else "",
        "relation_type": str(message.metadata.get("relation_type", "")) if message.metadata else "",
        "context_vector": _tensor_to_list(message.context),
    }


def _action_to_dict(action: PredictedGraphAction) -> dict[str, Any]:
    return {
        "action_type": action.action_type.value,
        "score": float(action.score),
        "effective_time": float(action.effective_time),
        "source_node_id": action.source_node_id,
        "target_node_id": action.target_node_id,
        "relation_type": action.relation_type,
        "role": action.role,
        "new_node_id": action.new_node_id,
        "signature": _action_signature(action),
    }


def _tensor_to_list(tensor: torch.Tensor | None) -> list[float]:
    if tensor is None:
        return []
    values = tensor.detach().float().flatten().cpu().tolist()
    return [round(float(value), 6) for value in values]


def _query_text_from_messages(messages: list[Message]) -> str:
    query_texts: list[str] = []
    for message in messages:
        if message.action != MessageAction.QUERY_ARRIVAL:
            continue
        if message.metadata:
            query_text = str(
                message.metadata.get("query_text")
                or message.metadata.get("raw_text")
                or ""
            ).strip()
            if query_text:
                query_texts.append(query_text)
    return "\n".join(dict.fromkeys(query_texts))


def _apply_context_updates(graph: TemporalGraph, step: EpisodeStep) -> None:
    for node_id, context in step.context_updates.items():
        if node_id in graph.nodes:
            graph.update_node_context(
                node_id,
                context,
                context_text=step.context_text_updates.get(node_id),
            )


def _apply_actions_to_graph(
    graph: TemporalGraph,
    actions: list[PredictedGraphAction],
    *,
    context_dim: int,
    device: str,
) -> None:
    for action in actions:
        if action.action_type == GraphActionType.CREATE_EDGE:
            if action.source_node_id is None or action.target_node_id is None:
                continue
            if action.source_node_id not in graph.nodes or action.target_node_id not in graph.nodes:
                continue
            if graph.has_active_edge(action.source_node_id, action.target_node_id, action.effective_time):
                continue
            graph.add_edge(
                TemporalEdge(
                    source_node_id=action.source_node_id,
                    target_node_id=action.target_node_id,
                    start_time=action.effective_time,
                    end_time=action.effective_time + 1.0,
                )
            )
        elif action.action_type == GraphActionType.REMOVE_EDGE:
            if action.source_node_id is None or action.target_node_id is None:
                continue
            graph.deactivate_edge(action.source_node_id, action.target_node_id, action.effective_time)
        elif action.action_type == GraphActionType.ADD_NODE:
            role = action.role or "new_role"
            node_id = action.new_node_id or graph.generate_node_id(role)
            if node_id not in graph.nodes:
                graph.add_node(
                    TemporalNode.build(
                        node_id=node_id,
                        role=role,
                        context=None,
                        context_dim=context_dim,
                        device=device,
                    )
                )


def _action_signature(action: PredictedGraphAction) -> str:
    if action.action_type in {GraphActionType.CREATE_EDGE, GraphActionType.REMOVE_EDGE}:
        relation = action.relation_type or ""
        return (
            f"{action.action_type.value}:"
            f"{action.source_node_id or ''}->{action.target_node_id or ''}:"
            f"{relation}"
        )
    if action.action_type == GraphActionType.ADD_NODE:
        return f"{action.action_type.value}:{action.role or ''}"
    return action.action_type.value


def _load_rich_log_records(log_path: str | Path) -> list[dict[str, Any]]:
    path = Path(log_path)
    if not path.exists():
        raise FileNotFoundError(f"Rich log not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            records.append(payload)
    return records


def _group_records_by_dataset(
    records: list[dict[str, Any]],
    *,
    max_samples: int,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        dataset_name = str(record.get("dataset_name") or "unknown")
        grouped.setdefault(dataset_name, []).append(record)
    selected: dict[str, list[dict[str, Any]]] = {}
    for dataset_name, items in grouped.items():
        if max_samples <= 0 or len(items) <= max_samples:
            selected[dataset_name] = list(items)
            continue
        sampled = list(items)
        random.Random(_stable_dataset_seed(dataset_name, seed)).shuffle(sampled)
        selected[dataset_name] = sampled[:max_samples]
    return selected


def _stable_dataset_seed(dataset_name: str, seed: int) -> int:
    digest = hashlib.sha256(f"{dataset_name}:{seed}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _label_for_record(record: dict[str, Any], *, label_mode: str) -> str:
    if label_mode == "action_signature":
        return str(record.get("target_action_signature") or "no_op")
    return str(record.get("target_action_type") or "no_op")


def _build_feature_matrix(
    records: list[dict[str, Any]],
    *,
    signals: tuple[str, ...],
    text_cache: dict[str, torch.Tensor],
    text_dim: int,
    max_graph_nodes: int,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    rows = []
    signal_dims: tuple[int, ...] | None = None
    for record in records:
        pieces: list[torch.Tensor] = []
        for signal in signals:
            if signal == "node_outputs":
                pieces.append(_node_outputs_vector(record, text_cache=text_cache, text_dim=text_dim))
            elif signal == "graph_structure":
                pieces.append(_graph_structure_vector(record, max_graph_nodes=max_graph_nodes))
            elif signal == "query":
                pieces.append(_query_vector(record, text_cache=text_cache, text_dim=text_dim))
            else:
                raise ValueError(f"Unknown signal: {signal}")
        feature = torch.cat(pieces, dim=0)
        if signal_dims is None:
            signal_dims = tuple(int(piece.numel()) for piece in pieces)
        norm = feature.norm(p=2)
        if float(norm.item()) > 0.0:
            feature = feature / norm
        rows.append(feature)
    return torch.stack(rows, dim=0), (signal_dims or tuple())


def _hash_text(text: str, dim: int) -> torch.Tensor:
    vector = torch.zeros(dim, dtype=torch.float32)
    if dim <= 0:
        raise ValueError("feature_dim must be positive.")
    normalized = str(text or "").strip().lower()
    if not normalized:
        return vector
    tokens = normalized.split()
    features: list[tuple[str, float]] = [(f"tok:{token}", 1.0) for token in tokens[:256]]
    features.extend((f"bigram:{left}|{right}", 0.75) for left, right in zip(tokens, tokens[1:256]))
    for feature, weight in features:
        if not feature:
            continue
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], "big") % dim
        sign = 1.0 if digest[8] % 2 == 0 else -1.0
        vector[index] += sign * weight
    norm = vector.norm(p=2)
    if float(norm.item()) > 0.0:
        vector = vector / norm
    return vector


def _build_text_embedding_cache(
    dataset_records: dict[str, list[dict[str, Any]]],
    *,
    sentence_transformer_model: str,
    sentence_transformer_batch_size: int,
    device: torch.device,
    fallback_dim: int,
) -> tuple[dict[str, torch.Tensor], int, str]:
    texts: list[str] = []
    for dataset_name in _ordered_dataset_names(dataset_records):
        for record in dataset_records[dataset_name]:
            query = str(record.get("query", "")).strip()
            if query:
                texts.append(query)
            for node in record.get("nodes", []) or []:
                if isinstance(node, dict):
                    node_text = _node_embedding_text(node)
                    if node_text:
                        texts.append(node_text)
    unique_texts = list(dict.fromkeys(texts))
    if not unique_texts:
        return {"": torch.zeros(fallback_dim, dtype=torch.float32)}, fallback_dim, "fallback_hash"
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(sentence_transformer_model, device=str(device))
        embeddings = model.encode(
            unique_texts,
            batch_size=max(int(sentence_transformer_batch_size), 1),
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if embeddings.ndim == 1:
            embeddings = embeddings.unsqueeze(0)
        text_dim = int(embeddings.size(1))
        cache: dict[str, torch.Tensor] = {"": torch.zeros(text_dim, dtype=torch.float32)}
        for text, embedding in zip(unique_texts, embeddings):
            cache[text] = embedding.detach().cpu().float()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return cache, text_dim, sentence_transformer_model
    except (ImportError, OSError, RuntimeError):
        cache = {"": torch.zeros(fallback_dim, dtype=torch.float32)}
        for text in unique_texts:
            cache[text] = _hash_text(text, fallback_dim)
        return cache, fallback_dim, f"fallback_hash_{fallback_dim}"


def _node_embedding_text(node: dict[str, Any]) -> str:
    output_text = str(node.get("output_text", "")).strip()
    if not output_text:
        return ""
    role = str(node.get("role", "")).strip()
    node_id = str(node.get("node_id", "")).strip()
    return f"role: {role}\nnode: {node_id}\noutput: {output_text}"


def _node_outputs_vector(
    record: dict[str, Any],
    *,
    text_cache: dict[str, torch.Tensor],
    text_dim: int,
) -> torch.Tensor:
    vectors: list[torch.Tensor] = []
    for node in record.get("nodes", []) or []:
        if not isinstance(node, dict):
            continue
        node_text = _node_embedding_text(node)
        if not node_text:
            continue
        vectors.append(text_cache.get(node_text, torch.zeros(text_dim, dtype=torch.float32)))
    if not vectors:
        return torch.zeros(text_dim, dtype=torch.float32)
    stacked = torch.stack(vectors, dim=0)
    mean_vector = stacked.mean(dim=0)
    norm = mean_vector.norm(p=2)
    if float(norm.item()) > 0.0:
        mean_vector = mean_vector / norm
    return mean_vector


def _query_vector(
    record: dict[str, Any],
    *,
    text_cache: dict[str, torch.Tensor],
    text_dim: int,
) -> torch.Tensor:
    query = str(record.get("query", "")).strip()
    vector = text_cache.get(query, torch.zeros(text_dim, dtype=torch.float32))
    norm = vector.norm(p=2)
    if float(norm.item()) > 0.0:
        vector = vector / norm
    return vector


def _graph_structure_vector(record: dict[str, Any], *, max_graph_nodes: int) -> torch.Tensor:
    graph = record.get("graph_structure") or {}
    adjacency = graph.get("adjacency") if isinstance(graph, dict) else None
    matrix = torch.zeros((max_graph_nodes, max_graph_nodes), dtype=torch.float32)
    if isinstance(adjacency, list):
        for row_index, row in enumerate(adjacency[:max_graph_nodes]):
            if not isinstance(row, list):
                continue
            for column_index, value in enumerate(row[:max_graph_nodes]):
                matrix[row_index, column_index] = float(value)
    vector = matrix.flatten()
    norm = vector.norm(p=2)
    if float(norm.item()) > 0.0:
        vector = vector / norm
    return vector


def _max_graph_nodes(records: list[dict[str, Any]]) -> int:
    max_nodes = 1
    for record in records:
        graph = record.get("graph_structure") or {}
        if not isinstance(graph, dict):
            continue
        adjacency = graph.get("adjacency") or []
        if isinstance(adjacency, list):
            max_nodes = max(max_nodes, len(adjacency))
    return max_nodes


def _ordered_dataset_names(dataset_records: dict[str, list[dict[str, Any]]]) -> list[str]:
    preferred_order = {"coding": 0, "research": 1, "werewolf": 2}
    return sorted(
        dataset_records,
        key=lambda name: (preferred_order.get(name, 99), name),
    )


def _split_train_val_indices(
    *,
    labels: list[int],
    train_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(labels) < 2:
        raise ValueError("Need at least 2 samples for a train/validation split.")
    groups: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(int(label), []).append(index)
    train_indices: list[int] = []
    val_indices: list[int] = []
    rng = random.Random(seed)
    for label in sorted(groups):
        items = list(groups[label])
        rng.shuffle(items)
        if len(items) == 1:
            train_indices.extend(items)
            continue
        train_count = max(1, min(len(items) - 1, int(round(len(items) * train_fraction))))
        train_indices.extend(items[:train_count])
        val_indices.extend(items[train_count:])
    if not val_indices:
        items = list(range(len(labels)))
        rng.shuffle(items)
        train_count = max(1, min(len(items) - 1, int(round(len(items) * train_fraction))))
        train_indices = items[:train_count]
        val_indices = items[train_count:]
    return (
        torch.tensor(sorted(train_indices), dtype=torch.long),
        torch.tensor(sorted(val_indices), dtype=torch.long),
    )


def _resolve_hidden_dims(
    *,
    input_dim: int,
    sample_count: int,
    hidden_dim: int | None,
    explicit_hidden_dims: tuple[int, ...] | None,
) -> tuple[int, ...]:
    if explicit_hidden_dims:
        dims = tuple(int(value) for value in explicit_hidden_dims if int(value) > 0)
        if dims:
            return dims
    if hidden_dim is not None and int(hidden_dim) > 0:
        first = int(hidden_dim)
        second = max(64, _round_hidden(first // 2, base=64))
        return (first, second)
    hidden_cap = min(2048, max(256, sample_count * 16))
    first = min(hidden_cap, max(256, _round_hidden(int(math.ceil(input_dim * 0.85)), base=64)))
    second = min(1024, max(128, _round_hidden(int(math.ceil(first * 0.5)), base=64)))
    if second >= first:
        second = max(64, first // 2)
    return (int(first), int(second))


def _resolve_signal_projection_dim(
    *,
    signal_dims: tuple[int, ...],
    sample_count: int,
) -> int:
    max_signal_dim = max(signal_dims) if signal_dims else 64
    base = max(128, min(512, _round_hidden(max_signal_dim, base=64)))
    if sample_count < 40:
        return min(base, 256)
    return base


def _round_hidden(value: int, *, base: int) -> int:
    if value <= 0:
        return base
    return int(math.ceil(value / base) * base)


def _resolve_batch_size(*, train_count: int, device: torch.device) -> int:
    if train_count <= 0:
        return 1
    if device.type == "cuda":
        return max(16, min(256, train_count))
    return max(8, min(128, train_count))


def _train_one_combination(
    *,
    dataset_name: str,
    features: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    val_indices: torch.Tensor,
    signals: tuple[str, ...],
    class_count: int,
    signal_dims: tuple[int, ...],
    signal_projection_dim: int,
    hidden_dims: tuple[int, ...],
    batch_size: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    repeat_count: int,
) -> MLPCombinationResult:
    effective_epochs = max(int(epochs), 1)
    repeat_count = max(int(repeat_count), 1)
    if class_count <= 1:
        val_acc = 1.0
        val_acc_std = 0.0
        best_epoch = 1
    else:
        run_accuracies: list[float] = []
        best_epochs: list[int] = []
        for run_index in range(repeat_count):
            run_seed = seed + run_index * 9973
            torch.manual_seed(run_seed)
            random.seed(run_seed)
            standardized = features.clone()
            train_mean = standardized.index_select(0, train_indices).mean(dim=0, keepdim=True)
            train_std = standardized.index_select(0, train_indices).std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
            standardized = (standardized - train_mean) / train_std
            x = standardized.to(device=device, non_blocking=True)
            y = labels.to(device=device, non_blocking=True)
            train_indices_device = train_indices.to(device=device, non_blocking=True)
            val_indices_device = val_indices.to(device=device, non_blocking=True)
            model = _FusionMLP(
                signal_dims=signal_dims,
                projection_dim=signal_projection_dim,
                hidden_dims=hidden_dims,
                class_count=class_count,
            ).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=2e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.6,
                patience=4,
                min_lr=1e-5,
            )
            scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
            train_targets = labels.index_select(0, train_indices)
            class_weights = _class_weights(train_targets, class_count=class_count)
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
            loader = DataLoader(
                TensorDataset(
                    standardized.index_select(0, train_indices),
                    labels.index_select(0, train_indices),
                ),
                batch_size=max(batch_size, 1),
                shuffle=True,
                drop_last=False,
                pin_memory=device.type == "cuda",
            )
            best_val_acc = -1.0
            best_val_loss = float("inf")
            best_epoch_value = 1
            best_state: dict[str, torch.Tensor] | None = None
            patience = max(10, min(25, effective_epochs // 3))
            stale_epochs = 0
            for epoch_index in range(effective_epochs):
                model.train()
                for batch_x, batch_y in loader:
                    batch_x = batch_x.to(device=device, non_blocking=True)
                    batch_y = batch_y.to(device=device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                        logits = model(batch_x)
                        loss = criterion(logits, batch_y)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                val_acc_candidate, val_loss_candidate = _accuracy_and_loss(
                    model=model,
                    features=x,
                    labels=y,
                    indices=val_indices_device,
                    criterion=criterion,
                )
                scheduler.step(val_loss_candidate)
                improved = (
                    val_acc_candidate > best_val_acc + 1e-6
                    or (
                        abs(val_acc_candidate - best_val_acc) <= 1e-6
                        and val_loss_candidate < best_val_loss - 1e-6
                    )
                )
                if improved:
                    best_val_acc = val_acc_candidate
                    best_val_loss = val_loss_candidate
                    best_epoch_value = epoch_index + 1
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                    if stale_epochs >= patience:
                        break
            if best_state is not None:
                model.load_state_dict(best_state)
            final_val_acc = _accuracy(model, x, y, val_indices_device)
            run_accuracies.append(final_val_acc)
            best_epochs.append(best_epoch_value)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        val_acc = float(sum(run_accuracies) / len(run_accuracies))
        val_acc_std = float(torch.tensor(run_accuracies, dtype=torch.float32).std(unbiased=False).item())
        best_epoch = int(round(sum(best_epochs) / len(best_epochs)))
    return MLPCombinationResult(
        dataset_name=dataset_name,
        combination="+".join(signals),
        signals=list(signals),
        sample_count=int(features.size(0)),
        train_count=int(train_indices.numel()),
        val_count=int(val_indices.numel()),
        class_count=class_count,
        input_dim=int(features.size(1)),
        hidden_dims=[int(value) for value in hidden_dims],
        batch_size=int(batch_size),
        val_accuracy=val_acc,
        val_accuracy_std=val_acc_std,
        run_count=repeat_count,
        best_epoch=best_epoch,
        signal_dims=[int(value) for value in signal_dims],
        signal_projection_dim=int(signal_projection_dim),
    )


@torch.no_grad()
def _accuracy(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
) -> float:
    if indices.numel() == 0:
        return 0.0
    model.eval()
    logits = model(features.index_select(0, indices))
    predictions = logits.argmax(dim=1)
    targets = labels.index_select(0, indices)
    return float((predictions == targets).float().mean().item())


@torch.no_grad()
def _accuracy_and_loss(
    *,
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    criterion: nn.Module,
) -> tuple[float, float]:
    if indices.numel() == 0:
        return 0.0, 0.0
    model.eval()
    logits = model(features.index_select(0, indices))
    targets = labels.index_select(0, indices)
    loss = criterion(logits, targets)
    predictions = logits.argmax(dim=1)
    accuracy = float((predictions == targets).float().mean().item())
    return accuracy, float(loss.item())


def _class_weights(labels: torch.Tensor, *, class_count: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=class_count).float().clamp_min(1.0)
    weights = counts.sum() / (counts * float(class_count))
    weights = weights / weights.mean().clamp_min(1e-6)
    return weights


def _write_mlp_report(path: Path, result: MLPTrainingResult) -> None:
    payload = asdict(result)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_mlp_csv(path: Path, dataset_results: list[MLPDatasetResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset_name",
                "combination",
                "signals",
                "sample_count",
                "train_count",
                "val_count",
                "class_count",
                "input_dim",
                "signal_dims",
                "signal_projection_dim",
                "hidden_dims",
                "batch_size",
                "val_accuracy",
                "val_accuracy_std",
                "run_count",
                "best_epoch",
            ]
        )
        for dataset_result in dataset_results:
            for result in dataset_result.combinations:
                writer.writerow(
                    [
                        result.dataset_name,
                        result.combination,
                        "|".join(result.signals),
                        result.sample_count,
                        result.train_count,
                        result.val_count,
                        result.class_count,
                        result.input_dim,
                        "|".join(str(value) for value in result.signal_dims),
                        result.signal_projection_dim,
                        "|".join(str(value) for value in result.hidden_dims),
                        result.batch_size,
                        f"{result.val_accuracy:.6f}",
                        f"{result.val_accuracy_std:.6f}",
                        result.run_count,
                        result.best_epoch,
                    ]
                )


def _write_accuracy_png(path: Path, dataset_results: list[MLPDatasetResult]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, PercentFormatter

    dataset_names = [result.dataset_name for result in dataset_results]
    if not dataset_names:
        raise ValueError("No dataset results available for chart generation.")
    combination_order = ["+".join(signals) for signals in INFO_COMBINATIONS]
    color_map = {
        "node_outputs": "#4E79A7",
        "graph_structure": "#A0A7B4",
        "query": "#59A14F",
        "node_outputs+graph_structure": "#F28E2B",
        "node_outputs+query": "#E15759",
        "graph_structure+query": "#B07AA1",
        "node_outputs+graph_structure+query": "#76B7B2",
    }
    legend_label_map = {
        "node_outputs": "Node",
        "graph_structure": "Graph",
        "query": "Query",
        "node_outputs+graph_structure": "Node + Graph",
        "node_outputs+query": "Node + Query",
        "graph_structure+query": "Graph + Query",
        "node_outputs+graph_structure+query": "All Signals",
    }
    dataset_lookup = {
        result.dataset_name: {item.combination: item for item in result.combinations}
        for result in dataset_results
    }
    x_positions = list(range(len(dataset_names)))
    all_values = [
        dataset_lookup[dataset_name][combination_name].val_accuracy
        for dataset_name in dataset_names
        for combination_name in combination_order
    ]
    raw_min = min(all_values)
    raw_max = max(all_values)
    y_min = max(0.0, math.floor((raw_min - 0.025) / 0.02) * 0.02)
    y_max = min(1.0, math.ceil((raw_max + 0.025) / 0.02) * 0.02)
    if y_max - y_min < 0.18:
        mid = (y_min + y_max) / 2.0
        y_min = max(0.0, mid - 0.10)
        y_max = min(1.0, mid + 0.10)

    fig, ax = plt.subplots(figsize=(12.4, 6.6), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    ax.spines["left"].set_color("#d4dbe5")
    ax.spines["bottom"].set_color("#d4dbe5")

    bar_width = 0.104
    edge_color = "#ffffff"
    for combo_index, combination_name in enumerate(combination_order):
        offsets = [
            position + (combo_index - (len(combination_order) - 1) / 2.0) * bar_width
            for position in x_positions
        ]
        values = [
            dataset_lookup[dataset_name][combination_name].val_accuracy
            for dataset_name in dataset_names
        ]
        ax.bar(
            offsets,
            values,
            width=bar_width * 0.92,
            label=legend_label_map.get(combination_name, combination_name),
            color=color_map.get(combination_name, "#4c72b0"),
            edgecolor=edge_color,
            linewidth=1.0,
            zorder=3,
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels([name.title() for name in dataset_names], fontsize=13, fontweight="semibold", color="#0f172a")
    ax.tick_params(axis="x", length=0, pad=8)
    ax.tick_params(axis="y", colors="#334155", labelsize=11.5)
    ax.set_ylim(y_min, y_max)
    tick_step = 0.02 if y_max - y_min <= 0.24 else 0.05
    ax.yaxis.set_major_locator(MultipleLocator(tick_step))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel("Validation Accuracy", fontsize=12.5, color="#334155")
    ax.set_xlabel("")
    ax.set_title("Validation Accuracy Across Datasets", fontsize=16.5, fontweight="semibold", color="#0f172a", pad=18)
    ax.grid(axis="y", linestyle=(0, (2, 3)), linewidth=0.8, alpha=0.22, color="#94a3b8", zorder=1)
    ax.set_axisbelow(True)
    ax.margins(x=0.05)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=4,
        frameon=False,
        fontsize=10.2,
        handlelength=1.6,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.98))
    fig.savefig(path, format="png", bbox_inches="tight")
    plt.close(fig)


def _short_combo_label(signals: list[str]) -> str:
    aliases = {
        "node_outputs": "node",
        "graph_structure": "graph",
        "query": "query",
    }
    return "+".join(aliases.get(signal, signal) for signal in signals)


__all__ = [
    "INFO_COMBINATIONS",
    "MLPCombinationResult",
    "MLPTrainingResult",
    "RichLogExportResult",
    "iter_rich_log_records",
    "train_mlp_on_rich_log",
    "write_rich_log",
]
