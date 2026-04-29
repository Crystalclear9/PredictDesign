from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import torch

from predictdesign import (
    BenchmarkEvaluator,
    BenchmarkEpisode,
    EpisodeStep,
    ExperimentConfig,
    GraphActionType,
    LLMApiGraphActionPredictor,
    PredictDesignSystem,
    TemporalEdge,
    TemporalNode,
)
from predictdesign.benchmark.multiagentbench import MultiAgentBenchAdapter
from predictdesign.benchmark.rich_log import train_mlp_on_rich_log, write_rich_log
from predictdesign.benchmark.trainer import BenchmarkTrainer
from predictdesign.messages import Message
from predictdesign.prediction import PredictedGraphAction
from predictdesign.state_update import MDPStateUpdater, build_state_updater


class PredictDesignTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(3)

    def _build_system(self, concurrent_mode: str = "sum", updater: str = "gru") -> PredictDesignSystem:
        config = ExperimentConfig(
            context_dim=6,
            hidden_dim=12,
            concurrent_update_mode=concurrent_mode,
            state_updater_type=updater,
            gnn_type="gcn",
            prediction_horizon=2,
            candidate_new_roles=("planner", "coder"),
        )
        system = PredictDesignSystem(config=config)
        system.initialize_graph(
            nodes=[
                TemporalNode.build("a", "planner", [1, 0, 0, 0, 0, 0], 6, "cpu"),
                TemporalNode.build("b", "coder", [0, 1, 0, 0, 0, 0], 6, "cpu"),
            ]
        )
        return system

    def test_state_updater_factory_supports_gru_and_mdp(self) -> None:
        gru = build_state_updater(
            "gru",
            context_dim=6,
            hidden_dim=12,
            latent_state_count=5,
            latent_action_count=3,
        )
        mdp = build_state_updater(
            "mdp",
            context_dim=6,
            hidden_dim=12,
            latent_state_count=5,
            latent_action_count=3,
        )
        self.assertEqual(gru.hidden_dim, 12)
        self.assertEqual(mdp.hidden_dim, 12)
        self.assertEqual(mdp.latent_action_count, 3)

    def test_concurrent_message_reduce_modes_are_switchable(self) -> None:
        sum_system = self._build_system(concurrent_mode="sum")
        mean_system = self._build_system(concurrent_mode="mean")
        messages = [
            Message.build_completion_message(
                time=1.0,
                source_node_id="a",
                target_node_id="b",
                context=[0.2] * 6,
                hidden_dim=12,
                context_dim=6,
            ),
            Message.build_completion_message(
                time=1.0,
                source_node_id="a",
                target_node_id="b",
                context=[0.4] * 6,
                hidden_dim=12,
                context_dim=6,
            ),
        ]
        sum_system.ingest_messages(messages)
        mean_system.ingest_messages(messages)
        sum_state = sum_system.ctdg.get_state("b")
        mean_state = mean_system.ctdg.get_state("b")
        self.assertEqual(sum_state.shape, mean_state.shape)
        self.assertFalse(torch.allclose(sum_state, mean_state))

    def test_predict_rollout_returns_configured_horizon(self) -> None:
        system = self._build_system()
        system.ingest_messages(
            [
                Message.build_query_message(
                    target_node_id="a",
                    time=0.0,
                    context=[0.1] * 6,
                    context_dim=6,
                    device="cpu",
                )
            ]
        )
        rollout = system.predict_next_steps(observation_time=1.0, steps=2)
        self.assertEqual(len(rollout.actions), 2)
        for action in rollout.actions:
            self.assertIn(
                action.action_type,
                {
                    GraphActionType.CREATE_EDGE,
                    GraphActionType.REMOVE_EDGE,
                    GraphActionType.ADD_NODE,
                    GraphActionType.NO_OP,
                },
            )

    def test_add_node_action_updates_both_graphs(self) -> None:
        system = self._build_system()
        action = system.predictor.predict_next_action(
            temporal_graph=system.temporal_graph,
            ctdg=system.ctdg,
            observation_time=1.0,
        )
        manual_add = action
        manual_add.action_type = GraphActionType.ADD_NODE
        manual_add.role = "planner"
        manual_add.source_node_id = None
        manual_add.target_node_id = None
        system.predictor.apply_action(
            action=manual_add,
            temporal_graph=system.temporal_graph,
            ctdg=system.ctdg,
        )
        self.assertIsNotNone(manual_add.new_node_id)
        self.assertIn(manual_add.new_node_id, system.temporal_graph.nodes)
        self.assertIn(manual_add.new_node_id, system.ctdg.current_states)

    def test_empty_initial_graph_can_still_predict(self) -> None:
        config = ExperimentConfig(
            context_dim=6,
            hidden_dim=12,
            prediction_horizon=1,
            candidate_new_roles=("planner", "coder"),
        )
        system = PredictDesignSystem(config=config)
        system.initialize_graph(nodes=[], edges=[])
        action = system.predictor.predict_next_action(
            temporal_graph=system.temporal_graph,
            ctdg=system.ctdg,
            observation_time=0.0,
        )
        self.assertIn(action.action_type, {GraphActionType.ADD_NODE, GraphActionType.NO_OP})

    def test_query_parser_initializes_graph_and_ctdg(self) -> None:
        config = ExperimentConfig(
            context_dim=6,
            hidden_dim=12,
            prediction_horizon=1,
            candidate_new_roles=("planner", "coder"),
        )
        system = PredictDesignSystem(config=config)
        parse_result = system.initialize_from_query(
            "Alice: planner, Bob(coder) collaborate on debugging and review.",
            inject_query_message=True,
        )
        self.assertEqual(sorted(node.node_id for node in parse_result.nodes), ["Alice", "Bob"])
        self.assertIn("Alice", system.temporal_graph.nodes)
        self.assertIn("Bob", system.temporal_graph.nodes)
        self.assertIn("Alice", system.ctdg.current_states)
        self.assertGreaterEqual(len(system.ctdg.state_history["Alice"]), 1)
        self.assertGreaterEqual(len(system.ctdg.state_history["Bob"]), 1)

    def test_temporal_edge_features_include_time_fields(self) -> None:
        system = self._build_system()
        system.add_edge("a", "b", start_time=1.0, end_time=5.0)
        features = system.temporal_graph.temporal_edge_features(
            time_value=3.0,
            node_order=["a", "b"],
        )
        self.assertEqual(features.shape, (2, 2, 4))
        self.assertGreater(float(features[0, 1, 0].item()), 0.0)
        self.assertGreater(float(features[0, 1, 1].item()), 0.0)
        self.assertGreater(float(features[0, 1, 2].item()), 0.0)
        self.assertGreater(float(features[0, 1, 3].item()), 0.0)

    def test_predictor_augments_edge_features_with_semantics(self) -> None:
        config = ExperimentConfig(
            context_dim=6,
            hidden_dim=12,
            temporal_edge_dim=9,
            candidate_new_roles=("planner", "coder"),
        )
        system = PredictDesignSystem(config=config)
        system.initialize_graph(
            nodes=[
                TemporalNode.build("a", "planner", [1, 1, 0, 0, 0, 0], 6, "cpu"),
                TemporalNode.build("b", "planner", [1, 1, 0, 0, 0, 0], 6, "cpu"),
            ],
            edges=[],
            structural_edges=[("a", "b")],
        )
        _, _, edge_features, _ = system.predictor.encode_graph(
            temporal_graph=system.temporal_graph,
            ctdg=system.ctdg,
            observation_time=0.0,
        )
        self.assertEqual(edge_features.shape[-1], 9)
        self.assertGreater(float(edge_features[0, 1, 5:].abs().sum().item()), 0.0)

    def test_rollout_apply_action_updates_ctdg_state_when_requested(self) -> None:
        system = self._build_system()
        previous_state = system.ctdg.get_state("b").clone()
        action = PredictedGraphAction(
            action_type=GraphActionType.CREATE_EDGE,
            score=1.0,
            effective_time=1.0,
            source_node_id="a",
            target_node_id="b",
            relation_type="communication",
        )
        system.predictor.apply_action(
            action=action,
            temporal_graph=system.temporal_graph,
            ctdg=system.ctdg,
            update_state=True,
        )
        next_state = system.ctdg.get_state("b")
        self.assertFalse(torch.allclose(previous_state, next_state))

    def test_relation_role_prior_prefers_werewolf_actions_for_wolf_source(self) -> None:
        config = ExperimentConfig(
            context_dim=6,
            hidden_dim=12,
            temporal_edge_dim=9,
            candidate_new_roles=("planner", "coder"),
        )
        system = PredictDesignSystem(config=config)
        system.initialize_graph(
            nodes=[
                TemporalNode.build("wolf_a", "wolf", [1, 0, 0, 0, 0, 0], 6, "cpu"),
                TemporalNode.build("villager_b", "villager", [0, 1, 0, 0, 0, 0], 6, "cpu"),
            ]
        )
        bundle = system.predictor.score_action_space(
            temporal_graph=system.temporal_graph,
            ctdg=system.ctdg,
            observation_time=1.0,
        )
        row = bundle.node_order.index("wolf_a")
        col = bundle.node_order.index("villager_b")
        relation_scores = bundle.relation_logits[row, col]
        werewolf_vote_idx = system.config.candidate_relation_types.index("werewolf_vote")
        seer_check_idx = system.config.candidate_relation_types.index("seer_check")
        self.assertGreater(
            float(relation_scores[werewolf_vote_idx].item()),
            float(relation_scores[seer_check_idx].item()),
        )

    def test_llm_api_predictor_parses_json_actions(self) -> None:
        def fake_completion(system_prompt: str, user_prompt: str, config: ExperimentConfig) -> str:
            self.assertIn("Return JSON only", user_prompt)
            return """
            {
              "predicted_count": 1,
              "actions": [
                {
                  "action_type": "create_edge",
                  "source_node_id": "a",
                  "target_node_id": "b",
                  "relation_type": "communication",
                  "score": 0.93
                }
              ]
            }
            """

        config = ExperimentConfig(
            context_dim=6,
            hidden_dim=12,
            gnn_type="llm_api",
            predictor_backend="llm_api",
        )
        predictor = LLMApiGraphActionPredictor(config=config, completion_fn=fake_completion)
        system = PredictDesignSystem(config=config, predictor=predictor)
        system.initialize_graph(
            nodes=[
                TemporalNode.build("a", "planner", [1, 0, 0, 0, 0, 0], 6, "cpu"),
                TemporalNode.build("b", "coder", [0, 1, 0, 0, 0, 0], 6, "cpu"),
            ]
        )
        action = system.predictor.predict_next_action(
            temporal_graph=system.temporal_graph,
            ctdg=system.ctdg,
            observation_time=1.0,
        )
        self.assertEqual(action.action_type, GraphActionType.CREATE_EDGE)
        self.assertEqual(action.source_node_id, "a")
        self.assertEqual(action.target_node_id, "b")
        self.assertEqual(action.relation_type, "communication")

    def test_llm_api_prompt_keeps_full_node_output_text(self) -> None:
        long_text = "alpha " * 400

        def fake_completion(system_prompt: str, user_prompt: str, config: ExperimentConfig) -> str:
            self.assertIn(long_text.strip(), user_prompt)
            self.assertIn("No prompt-side truncation has been applied", user_prompt)
            return '{"predicted_count": 0, "actions": []}'

        config = ExperimentConfig(
            context_dim=6,
            hidden_dim=12,
            gnn_type="llm_api",
            predictor_backend="llm_api",
        )
        predictor = LLMApiGraphActionPredictor(config=config, completion_fn=fake_completion)
        system = PredictDesignSystem(config=config, predictor=predictor)
        system.initialize_graph(
            nodes=[
                TemporalNode.build("a", "planner", [1, 0, 0, 0, 0, 0], 6, "cpu"),
                TemporalNode.build("b", "coder", [0, 1, 0, 0, 0, 0], 6, "cpu"),
            ]
        )
        system.update_node_context("a", [1, 0, 0, 0, 0, 0], text=long_text)
        system.predictor.predict_next_action(
            temporal_graph=system.temporal_graph,
            ctdg=system.ctdg,
            observation_time=1.0,
        )

    def test_ctdg_preserves_full_message_history(self) -> None:
        system = self._build_system()
        messages = [
            Message.build_completion_message(
                time=float(index),
                source_node_id="a",
                target_node_id="b",
                context=[0.1] * 6,
                hidden_dim=12,
                context_dim=6,
            )
            for index in range(150)
        ]
        system.ingest_messages(messages)
        self.assertEqual(len(system.ctdg.message_history), 150)

    def test_benchmark_evaluator_supports_llm_api_backend(self) -> None:
        def fake_completion(system_prompt: str, user_prompt: str, config: ExperimentConfig) -> str:
            self.assertIn("current graph structure", user_prompt)
            self.assertIn("current output summary", user_prompt)
            return """
            {
              "predicted_count": 1,
              "actions": [
                {
                  "action_type": "create_edge",
                  "source_node_id": "a",
                  "target_node_id": "b",
                  "relation_type": "communication",
                  "score": 0.8
                }
              ]
            }
            """

        evaluator = BenchmarkEvaluator(
            context_dim=6,
            hidden_dim=12,
            train_epochs=0,
            llm_completion_fn=fake_completion,
        )
        episodes = [
            BenchmarkEpisode(
                episode_id="llm-toy",
                dataset_name="toyset",
                initial_nodes=[
                    TemporalNode.build("a", "planner", [1, 0, 0, 0, 0, 0], 6, "cpu"),
                    TemporalNode.build("b", "coder", [0, 1, 0, 0, 0, 0], 6, "cpu"),
                ],
                initial_edges=[],
                steps=[
                    EpisodeStep(
                        observation_time=1.0,
                        messages=[],
                        ground_truth_action=PredictedGraphAction(
                            action_type=GraphActionType.NO_OP,
                            score=1.0,
                            effective_time=1.0,
                        ),
                        observed_actions=[],
                    ),
                    EpisodeStep(
                        observation_time=2.0,
                        messages=[],
                        ground_truth_action=PredictedGraphAction(
                            action_type=GraphActionType.NO_OP,
                            score=1.0,
                            effective_time=2.0,
                        ),
                        observed_actions=[],
                        valid_next_actions=[
                            PredictedGraphAction(
                                action_type=GraphActionType.CREATE_EDGE,
                                score=1.0,
                                effective_time=2.0,
                                source_node_id="a",
                                target_node_id="b",
                                relation_type="communication",
                            )
                        ],
                    ),
                ],
            )
        ]
        results = evaluator.evaluate_dataset("toyset", episodes, gnn_types=("llm_api",))
        self.assertEqual(len(results), 1)
        self.assertTrue(all(item.gnn_type == "llm_api" for item in results))
        self.assertTrue(all(item.message_reduce == "llm_api" for item in results))
        self.assertTrue(all(item.state_updater == "llm_api" for item in results))

    def test_benchmark_evaluator_reports_multi_hit_k(self) -> None:
        def fake_completion(system_prompt: str, user_prompt: str, config: ExperimentConfig) -> str:
            return """
            {
              "predicted_count": 2,
              "actions": [
                {
                  "action_type": "create_edge",
                  "source_node_id": "a",
                  "target_node_id": "c",
                  "relation_type": "communication",
                  "score": 0.9
                },
                {
                  "action_type": "create_edge",
                  "source_node_id": "a",
                  "target_node_id": "b",
                  "relation_type": "communication",
                  "score": 0.8
                }
              ]
            }
            """

        evaluator = BenchmarkEvaluator(
            context_dim=6,
            hidden_dim=12,
            train_epochs=0,
            hit_k_values=(1, 2),
            llm_completion_fn=fake_completion,
        )
        episodes = [
            BenchmarkEpisode(
                episode_id="llm-hitk",
                dataset_name="toyset",
                initial_nodes=[
                    TemporalNode.build("a", "planner", [1, 0, 0, 0, 0, 0], 6, "cpu"),
                    TemporalNode.build("b", "coder", [0, 1, 0, 0, 0, 0], 6, "cpu"),
                    TemporalNode.build("c", "critic", [0, 0, 1, 0, 0, 0], 6, "cpu"),
                ],
                initial_edges=[],
                steps=[
                    EpisodeStep(
                        observation_time=1.0,
                        messages=[],
                        ground_truth_action=PredictedGraphAction(
                            action_type=GraphActionType.NO_OP,
                            score=1.0,
                            effective_time=1.0,
                        ),
                        observed_actions=[],
                    ),
                    EpisodeStep(
                        observation_time=2.0,
                        messages=[],
                        ground_truth_action=PredictedGraphAction(
                            action_type=GraphActionType.NO_OP,
                            score=1.0,
                            effective_time=2.0,
                        ),
                        observed_actions=[],
                        valid_next_actions=[
                            PredictedGraphAction(
                                action_type=GraphActionType.CREATE_EDGE,
                                score=1.0,
                                effective_time=2.0,
                                source_node_id="a",
                                target_node_id="b",
                                relation_type="communication",
                            )
                        ],
                    ),
                ],
            )
        ]
        result = evaluator.evaluate_dataset("toyset", episodes, gnn_types=("llm_api",))[0]
        self.assertEqual(result.hit_ks, (1, 2))
        self.assertEqual(result.hit_at_k["1"], 0.0)
        self.assertEqual(result.hit_at_k["2"], 1.0)
        self.assertEqual(result.accuracy, 0.0)

    def test_relation_type_is_preserved_in_ground_truth_actions(self) -> None:
        adapter = MultiAgentBenchAdapter(context_dim=6, hidden_dim=12, device="cpu")
        message = adapter._build_completion_message(
            source_node_id="a",
            target_node_id="b",
            time_value=1.0,
            text="banishment_vote:a->b",
        )
        message.metadata["raw_text"] = "banishment_vote:a->b"
        actions = adapter._message_ground_truths([message], 1.0)
        self.assertEqual(actions[0].relation_type, "banishment_vote")

    def test_evaluator_edge_match_checks_relation_type(self) -> None:
        evaluator = BenchmarkEvaluator(context_dim=6, hidden_dim=12, train_epochs=0)
        predicted = PredictedGraphAction(
            action_type=GraphActionType.CREATE_EDGE,
            score=1.0,
            effective_time=1.0,
            source_node_id="a",
            target_node_id="b",
            relation_type="communication",
        )
        expected = PredictedGraphAction(
            action_type=GraphActionType.CREATE_EDGE,
            score=1.0,
            effective_time=1.0,
            source_node_id="a",
            target_node_id="b",
            relation_type="banishment_vote",
        )
        self.assertFalse(evaluator._actions_match(predicted, expected))

    def test_mdp_updater_exposes_transition_summary(self) -> None:
        updater = MDPStateUpdater(
            context_dim=6,
            hidden_dim=12,
            latent_state_count=5,
            latent_action_count=3,
        )
        next_state, summary = updater.step(
            previous_state=torch.zeros(12),
            node_context=torch.ones(6),
            aggregated_message=torch.full((12,), 0.5),
        )
        self.assertEqual(next_state.shape, (12,))
        self.assertEqual(summary.action_probs.shape, (3,))
        self.assertEqual(summary.state_probs.shape, (5,))
        self.assertEqual(summary.transition_matrix.shape, (3, 5))

    def test_benchmark_evaluator_runs_all_module_combinations(self) -> None:
        evaluator = BenchmarkEvaluator(
            context_dim=6,
            hidden_dim=12,
            candidate_new_roles=("planner", "coder"),
            train_epochs=3,
        )
        episodes = [
            BenchmarkEpisode(
                episode_id="toy",
                dataset_name="toyset",
                initial_nodes=[
                    TemporalNode.build("a", "planner", [1, 0, 0, 0, 0, 0], 6, "cpu"),
                    TemporalNode.build("b", "coder", [0, 1, 0, 0, 0, 0], 6, "cpu"),
                ],
                initial_edges=[],
                steps=[
                    EpisodeStep(
                        observation_time=1.0,
                        messages=[
                            Message.build_completion_message(
                                time=1.0,
                                source_node_id="a",
                                target_node_id="b",
                                context=[0.2] * 6,
                                hidden_dim=12,
                                context_dim=6,
                                device="cpu",
                            )
                        ],
                        ground_truth_action=PredictedGraphAction(
                            action_type=GraphActionType.CREATE_EDGE,
                            score=1.0,
                            effective_time=1.0,
                            source_node_id="a",
                            target_node_id="b",
                        ),
                    ),
                    EpisodeStep(
                        observation_time=2.0,
                        messages=[],
                        ground_truth_action=PredictedGraphAction(
                            action_type=GraphActionType.NO_OP,
                            score=1.0,
                            effective_time=2.0,
                        ),
                    )
                ],
            )
        ]
        results = evaluator.evaluate_dataset("toyset", episodes)
        self.assertEqual(len(results), 12)
        self.assertTrue(all(item.total_steps == 1 for item in results))

    def test_research_adapter_uses_next_step_as_supervision(self) -> None:
        adapter = MultiAgentBenchAdapter(context_dim=6, hidden_dim=12, device="cpu")
        payload = {
            "task": "investigate",
            "agent_profiles": {
                "agent1": {"profile": "planner"},
                "agent2": {"profile": "coder"},
            },
            "iterations": [
                {
                    "communications": ["From agent1 to agent2: plan first"],
                    "task_results": [],
                },
                {
                    "communications": ["From agent2 to agent1: code back"],
                    "task_results": [],
                },
            ],
        }
        episode = adapter._research_payload_to_episode(payload, line_number=1)
        first_step = episode.steps[0]
        second_step = episode.steps[1]
        self.assertEqual(
            [(action.source_node_id, action.target_node_id) for action in first_step.observed_actions],
            [("agent1", "agent2")],
        )
        self.assertEqual(
            [(action.source_node_id, action.target_node_id) for action in first_step.supervision_actions],
            [("agent2", "agent1")],
        )
        self.assertEqual(len(episode.initial_structural_edges), 2)

    def test_future_rollout_targets_use_supervision_actions(self) -> None:
        trainer = BenchmarkTrainer()
        episode = BenchmarkEpisode(
            episode_id="future-supervision",
            dataset_name="toyset",
            initial_nodes=[],
            initial_edges=[],
            steps=[
                EpisodeStep(
                    observation_time=1.0,
                    messages=[],
                    ground_truth_action=PredictedGraphAction(
                        action_type=GraphActionType.NO_OP,
                        score=1.0,
                        effective_time=1.0,
                    ),
                    observed_actions=[],
                ),
                EpisodeStep(
                    observation_time=2.0,
                    messages=[],
                    ground_truth_action=PredictedGraphAction(
                        action_type=GraphActionType.NO_OP,
                        score=1.0,
                        effective_time=2.0,
                    ),
                    observed_actions=[
                        PredictedGraphAction(
                            action_type=GraphActionType.NO_OP,
                            score=1.0,
                            effective_time=2.0,
                        )
                    ],
                    valid_next_actions=[
                        PredictedGraphAction(
                            action_type=GraphActionType.CREATE_EDGE,
                            score=1.0,
                            effective_time=2.0,
                            source_node_id="a",
                            target_node_id="b",
                        )
                    ],
                ),
            ],
        )
        targets = trainer._future_rollout_targets(episode, step_index=0, horizon=1)
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0][1][0].action_type, GraphActionType.CREATE_EDGE)

    def test_evaluator_counts_any_valid_parallel_action_as_correct(self) -> None:
        evaluator = BenchmarkEvaluator(context_dim=6, hidden_dim=12, train_epochs=0)
        predicted = PredictedGraphAction(
            action_type=GraphActionType.CREATE_EDGE,
            score=1.0,
            effective_time=1.0,
            source_node_id="a",
            target_node_id="b",
        )
        valid_actions = [
            PredictedGraphAction(
                action_type=GraphActionType.CREATE_EDGE,
                score=1.0,
                effective_time=1.0,
                source_node_id="c",
                target_node_id="d",
            ),
            PredictedGraphAction(
                action_type=GraphActionType.CREATE_EDGE,
                score=1.0,
                effective_time=1.0,
                source_node_id="a",
                target_node_id="b",
            ),
        ]
        self.assertTrue(evaluator._actions_match_any(predicted, valid_actions))

    def test_werewolf_builder_includes_seer_and_drops_self_target_edges(self) -> None:
        adapter = MultiAgentBenchAdapter(context_dim=6, hidden_dim=12, device="cpu")
        messages = adapter._build_werewolf_night_messages(
            night_event={
                "guard_action": "seer_a",
                "werewolf_action": {"final_target": "villager_a", "attack_successful": True},
                "player_dead_tonight": ["villager_a"],
                "witch_action": {"action": "none", "target": None},
            },
            players={
                "seer_a": {
                    "status": {
                        "check_history": {
                            "Night 1": {"player": "wolf_a", "result": "werewolf"}
                        }
                    }
                }
            },
            role_by_player={
                "wolf_a": "wolf",
                "wolf_b": "wolf",
                "seer_a": "seer",
                "villager_a": "villager",
            },
            known_players={"wolf_a", "wolf_b", "seer_a", "villager_a"},
            round_targets={"wolf_a": "villager_a", "wolf_b": "villager_a"},
            night_index=1,
            time_value=1.0,
        )
        pairs = [(message.source_node_id, message.target_node_id) for message in messages]
        self.assertIn(("seer_a", "wolf_a"), pairs)
        self.assertIn(("wolf_a", "villager_a"), pairs)
        self.assertNotIn(("seer_a", "seer_a"), pairs)

    def test_rich_log_export_includes_query_nodes_and_graph(self) -> None:
        adapter = MultiAgentBenchAdapter(context_dim=6, hidden_dim=12, device="cpu")
        episode = adapter._research_payload_to_episode(
            {
                "task": "investigate graph logging",
                "agent_profiles": {
                    "agent1": {"profile": "planner"},
                    "agent2": {"profile": "coder"},
                },
                "iterations": [
                    {
                        "communications": ["From agent1 to agent2: prepare plan"],
                        "task_results": [{"agent_id": "agent1", "result": "plan output"}],
                    },
                    {
                        "communications": ["From agent2 to agent1: implement"],
                        "task_results": [{"agent_id": "agent2", "result": "code output"}],
                    },
                ],
            },
            line_number=1,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "rich.jsonl"
            result = write_rich_log(log_path, [episode], context_dim=6)
            self.assertEqual(result.record_count, 2)
            records = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
            ]
        self.assertIn("investigate graph logging", records[0]["query"])
        self.assertIn("nodes", records[0])
        self.assertIn("graph_structure", records[0])
        self.assertIn("target_actions", records[0])
        self.assertGreaterEqual(records[0]["graph_structure"]["node_count"], 2)

    def test_rich_log_mlp_writes_reports_for_all_signal_combinations(self) -> None:
        adapter = MultiAgentBenchAdapter(context_dim=6, hidden_dim=12, device="cpu")
        episodes = [
            adapter._research_payload_to_episode(
                {
                    "task": f"query {index}",
                    "agent_profiles": {
                        "agent1": {"profile": "planner"},
                        "agent2": {"profile": "coder"},
                    },
                    "iterations": [
                        {
                            "communications": ["From agent1 to agent2: first"],
                            "task_results": [{"agent_id": "agent1", "result": "alpha"}],
                        },
                        {
                            "communications": ["From agent2 to agent1: second"],
                            "task_results": [{"agent_id": "agent2", "result": "beta"}],
                        },
                    ],
                },
                line_number=index,
            )
            for index in range(1, 3)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "rich.jsonl"
            out_dir = Path(tmpdir) / "mlp"
            write_rich_log(log_path, episodes, context_dim=6)
            result = train_mlp_on_rich_log(
                log_path,
                out_dir,
                max_samples=4,
                feature_dim=16,
                hidden_dim=8,
                epochs=2,
                sentence_transformer_model="__missing_sentence_transformer_model__",
            )
            self.assertEqual(len(result.combinations), 7)
            self.assertEqual(len(result.datasets), 1)
            self.assertTrue(Path(result.report_path).exists())
            self.assertTrue(Path(result.csv_path).exists())
            self.assertTrue(Path(result.chart_path).exists())


if __name__ == "__main__":
    unittest.main()
