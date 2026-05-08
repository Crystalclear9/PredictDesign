from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predictdesign import (
    ACGNapAdapter,
    ExperimentConfig,
    PredictDesignSystem,
    PredictedGraphAction,
    load_acg_nap_candidate_corpus,
)
from predictdesign.paths import ACG_NAP_ROOT, RESULTS_ROOT
from predictdesign.benchmark.trainer import BenchmarkTrainer, bootstrap_few_shot_transition_memory


def _multi_target_log_loss(logits: torch.Tensor, target_indices: list[int]) -> torch.Tensor:
    if not target_indices:
        return logits.new_tensor(0.0)
    log_probs = F.log_softmax(logits, dim=0)
    index_tensor = torch.tensor(target_indices, dtype=torch.long, device=logits.device)
    return -torch.logsumexp(log_probs.index_select(0, index_tensor), dim=0)


@dataclass(slots=True)
class CandidateEvalResult:
    dataset_name: str
    message_reduce: str
    state_updater: str
    gnn_type: str
    total_steps: int
    hit_ks: tuple[int, ...]
    hit_counts: dict[str, int]
    hit_at_k: dict[str, float]
    train_episode_count: int
    eval_episode_count: int


class ACGNapCandidateTrainer:
    def __init__(
        self,
        epochs: int = 5,
        learning_rate: float = 5e-3,
        weight_decay: float = 1e-4,
        seed: int = 7,
        hit_k_values: tuple[int, ...] = (1, 3, 5),
    ) -> None:
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.seed = seed
        self.hit_k_values = hit_k_values

    def fit(self, system: PredictDesignSystem, episodes) -> None:
        if episodes:
            bootstrap_few_shot_transition_memory(system, episodes)
        if not episodes or self.epochs <= 0:
            return
        torch.manual_seed(self.seed)
        optimizer = torch.optim.AdamW(
            system.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        system.train()
        for epoch_idx in range(self.epochs):
            shuffled = list(episodes)
            random.Random(self.seed + epoch_idx).shuffle(shuffled)
            for episode in shuffled:
                self._fit_episode(system, episode, optimizer)

    def _fit_episode(self, system: PredictDesignSystem, episode, optimizer) -> None:
        system.initialize_graph(
            nodes=episode.initial_nodes,
            edges=episode.initial_edges,
            structural_edges=episode.initial_structural_edges,
            graph_context_text=episode.initial_graph_context_text,
            structural_edge_metadata=episode.initial_structural_edge_metadata,
        )
        optimizer.zero_grad(set_to_none=True)
        episode_loss = next(system.parameters()).new_tensor(0.0)
        contributing_steps = 0
        for step in episode.steps:
            self._apply_context_updates(system, step)
            system.ingest_messages(step.messages)
            score_vector, target_indices, candidate_actions = self._candidate_score_vector(system, step)
            if score_vector is None or not target_indices:
                self._detach_ctdg_state(system)
                continue
            episode_loss = episode_loss + _multi_target_log_loss(score_vector, target_indices)
            contributing_steps += 1
            self._detach_ctdg_state(system)
        if contributing_steps <= 0:
            return
        normalized_loss = episode_loss / float(contributing_steps)
        normalized_loss.backward()
        torch.nn.utils.clip_grad_norm_(system.parameters(), max_norm=1.0)
        optimizer.step()

    @torch.no_grad()
    def evaluate(self, system: PredictDesignSystem, episodes, dataset_name: str) -> CandidateEvalResult:
        previous_mode = system.training
        system.eval()
        total_steps = 0
        hit_counts = {str(hit_k): 0 for hit_k in self.hit_k_values}
        for episode in episodes:
            system.initialize_graph(
                nodes=episode.initial_nodes,
                edges=episode.initial_edges,
                structural_edges=episode.initial_structural_edges,
                graph_context_text=episode.initial_graph_context_text,
                structural_edge_metadata=episode.initial_structural_edge_metadata,
            )
            for step in episode.steps:
                self._apply_context_updates(system, step)
                system.ingest_messages(step.messages)
                score_vector, target_indices, candidate_actions = self._candidate_score_vector(system, step)
                if score_vector is None or not target_indices or not candidate_actions:
                    continue
                ranked_indices = torch.argsort(score_vector, descending=True).tolist()
                for hit_k in self.hit_k_values:
                    top_k = ranked_indices[:hit_k]
                    if any(index in target_indices for index in top_k):
                        hit_counts[str(hit_k)] += 1
                total_steps += 1
        if previous_mode:
            system.train()
        hit_at_k = {
            key: (value / total_steps) if total_steps else 0.0
            for key, value in hit_counts.items()
        }
        return CandidateEvalResult(
            dataset_name=dataset_name,
            message_reduce=system.config.concurrent_update_mode,
            state_updater=system.config.state_updater_type,
            gnn_type=system.config.gnn_type,
            total_steps=total_steps,
            hit_ks=self.hit_k_values,
            hit_counts=hit_counts,
            hit_at_k=hit_at_k,
            train_episode_count=0,
            eval_episode_count=len(episodes),
        )

    def _candidate_score_vector(
        self,
        system: PredictDesignSystem,
        step,
    ) -> tuple[torch.Tensor | None, list[int], list[PredictedGraphAction]]:
        candidate_actions = step.candidate_actions or step.observed_actions or [step.ground_truth_action]
        if not candidate_actions:
            return None, [], []
        bundle = system.predictor.score_action_space(
            temporal_graph=system.temporal_graph,
            ctdg=system.ctdg,
            observation_time=step.observation_time,
            prediction_context=step.prediction_context,
        )
        if not bundle.node_order:
            return None, [], []
        if bundle.candidate_actions and bundle.candidate_scores is not None:
            filtered_actions = list(bundle.candidate_actions)
            target_indices = [
                idx
                for idx, action in enumerate(filtered_actions)
                if self._action_matches_any(action, step.observed_actions or [step.ground_truth_action])
            ]
            return bundle.candidate_scores, target_indices, filtered_actions
        relation_index = {
            relation_type: index
            for index, relation_type in enumerate(system.config.candidate_relation_types)
        }
        scores: list[torch.Tensor] = []
        filtered_actions: list[PredictedGraphAction] = []
        target_indices: list[int] = []
        for action in candidate_actions:
            if (
                action.source_node_id is None
                or action.target_node_id is None
                or action.source_node_id not in bundle.node_order
                or action.target_node_id not in bundle.node_order
                or action.relation_type not in relation_index
            ):
                continue
            row = bundle.node_order.index(action.source_node_id)
            col = bundle.node_order.index(action.target_node_id)
            if (
                not system.config.allow_self_loop_prediction
                and action.source_node_id == action.target_node_id
            ):
                continue
            relation_idx = relation_index[action.relation_type]
            score = bundle.create_scores[row, col] + bundle.relation_logits[row, col, relation_idx]
            filtered_actions.append(action)
            scores.append(score)
        if not scores:
            return None, [], []
        score_vector = torch.stack(scores)
        for idx, action in enumerate(filtered_actions):
            if self._action_matches_any(action, step.observed_actions or [step.ground_truth_action]):
                target_indices.append(idx)
        return score_vector, target_indices, filtered_actions

    def _apply_context_updates(self, system: PredictDesignSystem, step) -> None:
        for node_id, context in step.context_updates.items():
            system.update_node_context(
                node_id,
                context,
                text=step.context_text_updates.get(node_id),
            )

    def _detach_ctdg_state(self, system: PredictDesignSystem) -> None:
        system.ctdg.current_states = {
            node_id: state.detach()
            for node_id, state in system.ctdg.current_states.items()
        }

    def _action_matches_any(
        self,
        action: PredictedGraphAction,
        expected_actions: list[PredictedGraphAction],
    ) -> bool:
        return any(
            action.action_type == expected.action_type
            and action.source_node_id == expected.source_node_id
            and action.target_node_id == expected.target_node_id
            and action.relation_type == expected.relation_type
            for expected in expected_actions
        )


def build_system(args, candidate_roles, candidate_relations) -> PredictDesignSystem:
    config = ExperimentConfig(
        context_dim=args.context_dim,
        hidden_dim=args.hidden_dim,
        concurrent_update_mode=args.message_reduce_mode,
        state_updater_type=args.state_updater,
        gnn_type=args.gnn_type,
        candidate_new_roles=candidate_roles,
        candidate_relation_types=candidate_relations,
        allow_self_loop_prediction=args.allow_self_loops,
        device=args.device,
        sentence_transformer_path=args.sentence_transformer_path,
        sentence_transformer_dim=args.sentence_transformer_dim,
        sentence_transformer_freeze=args.sentence_transformer_freeze,
    )
    return PredictDesignSystem(config=config)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate PredictDesign on acg_nap as a current-step candidate-ranking task."
    )
    parser.add_argument("--acg-nap-root", type=str, default=str(ACG_NAP_ROOT))
    parser.add_argument("--report-path", type=str, default=str(RESULTS_ROOT / "acg_nap" / "candidate_report.json"))
    parser.add_argument("--split-summary-path", type=str, default=str(RESULTS_ROOT / "acg_nap" / "candidate_split_summary.json"))
    parser.add_argument("--cleaning-summary-path", type=str, default=str(RESULTS_ROOT / "acg_nap" / "candidate_cleaning_summary.json"))
    parser.add_argument("--context-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--train-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--gnn-type",
        type=str,
        default="hybrid",
        choices=["gcn", "graphsage", "gat", "relational_transformer", "hybrid"],
    )
    parser.add_argument("--state-updater", type=str, default="gru")
    parser.add_argument("--message-reduce-mode", type=str, default="attention")
    parser.add_argument("--sentence-transformer-path", type=str, required=True)
    parser.add_argument("--sentence-transformer-dim", type=int, default=384)
    parser.add_argument("--sentence-transformer-freeze", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-self-loops", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-graph-profile-chars", type=int, default=240)
    parser.add_argument("--max-node-text-chars", type=int, default=480)
    parser.add_argument("--max-files-per-dataset", type=int, default=0)
    args = parser.parse_args()

    adapter = ACGNapAdapter(
        context_dim=args.context_dim,
        hidden_dim=args.hidden_dim,
        device=args.device,
        max_graph_profile_chars=args.max_graph_profile_chars,
        max_node_text_chars=args.max_node_text_chars,
    )
    corpus = load_acg_nap_candidate_corpus(
        args.acg_nap_root,
        adapter,
        max_files_per_dataset=(args.max_files_per_dataset or None),
    )
    split_trainer = BenchmarkTrainer(train_fraction=args.train_fraction, seed=args.seed)
    dataset_splits = corpus.dataset_splits(split_trainer)
    combined_split = corpus.combined_split(split_trainer)

    system = build_system(
        args,
        candidate_roles=corpus.role_types or ("planner", "researcher"),
        candidate_relations=corpus.relation_types or ("activate", "delegate", "delegate_return", "retry"),
    )
    trainer = ACGNapCandidateTrainer(
        epochs=args.train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    trainer.fit(system, combined_split.train_episodes)

    report_records: list[CandidateEvalResult] = []
    all_result = trainer.evaluate(system, combined_split.eval_episodes, dataset_name="acg_nap_all")
    all_result.train_episode_count = len(combined_split.train_episodes)
    report_records.append(all_result)
    for dataset_name, split in dataset_splits.items():
        item = trainer.evaluate(system, split.eval_episodes, dataset_name=dataset_name)
        item.train_episode_count = len(combined_split.train_episodes)
        report_records.append(item)

    split_summary = {
        "acg_nap_root": str(corpus.root_path),
        "train_fraction": args.train_fraction,
        "seed": args.seed,
        "candidate_relation_types": list(corpus.relation_types),
        "candidate_new_roles": list(corpus.role_types),
        "combined": {
            "train_episode_count": len(combined_split.train_episodes),
            "eval_episode_count": len(combined_split.eval_episodes),
            "train_episode_ids": [episode.episode_id for episode in combined_split.train_episodes],
            "eval_episode_ids": [episode.episode_id for episode in combined_split.eval_episodes],
        },
        "datasets": {
            dataset_name: {
                "source_count": corpus.datasets[dataset_name].source_count,
                "episode_count": corpus.datasets[dataset_name].episode_count,
                "train_episode_count": len(split.train_episodes),
                "eval_episode_count": len(split.eval_episodes),
                "train_episode_ids": [episode.episode_id for episode in split.train_episodes],
                "eval_episode_ids": [episode.episode_id for episode in split.eval_episodes],
            }
            for dataset_name, split in dataset_splits.items()
        },
    }

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps([asdict(item) for item in report_records], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    split_summary_path = Path(args.split_summary_path)
    split_summary_path.parent.mkdir(parents=True, exist_ok=True)
    split_summary_path.write_text(json.dumps(split_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    cleaning_summary_path = Path(args.cleaning_summary_path)
    cleaning_summary_path.parent.mkdir(parents=True, exist_ok=True)
    cleaning_summary_path.write_text(json.dumps(corpus.cleaning_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"report={report_path}")
    print(f"split_summary={split_summary_path}")
    print(f"cleaning_summary={cleaning_summary_path}")
    print(f"combined_split train={len(combined_split.train_episodes)} eval={len(combined_split.eval_episodes)}")
    for dataset_name, split in dataset_splits.items():
        print(f"{dataset_name} train={len(split.train_episodes)} eval={len(split.eval_episodes)}")
    for item in report_records:
        print(
            f"{item.dataset_name}\t{item.message_reduce}\t{item.state_updater}\t{item.gnn_type}\t"
            f"hit@1={item.hit_at_k.get('1', 0.0):.4f}\t"
            f"hit@3={item.hit_at_k.get('3', 0.0):.4f}\t"
            f"hit@5={item.hit_at_k.get('5', 0.0):.4f}"
        )


if __name__ == "__main__":
    main()

