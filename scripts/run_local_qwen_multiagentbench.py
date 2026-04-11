from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable
from contextlib import contextmanager

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predictdesign.benchmark.evaluator import BenchmarkEvaluator
from predictdesign.benchmark.multiagentbench import MultiAgentBenchAdapter
from predictdesign.config import LLMApiConfig


REPO_ROOT = PROJECT_ROOT.parent
MULTIAGENTBENCH_ROOT = REPO_ROOT / "MultiAgentBench"
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "Qwen3.5-9B"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "local_qwen_multiagentbench"


@contextmanager
def _pushd(path: Path):
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return payload


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)


def _run_command(command: list[str], cwd: Path, fast_mode: bool = True) -> None:
    print("$", " ".join(command))
    env = os.environ.copy()
    if fast_mode:
        env.setdefault("PREDICTDESIGN_FAST_TOOL_CALLS", "1")
        env.setdefault("PREDICTDESIGN_LOCAL_MAX_NEW_TOKENS", "96")
        env.setdefault("PREDICTDESIGN_FAST_LOCAL_PLANNER", "1")
    else:
        env.pop("PREDICTDESIGN_FAST_TOOL_CALLS", None)
        env.pop("PREDICTDESIGN_LOCAL_MAX_NEW_TOKENS", None)
        env.pop("PREDICTDESIGN_FAST_LOCAL_PLANNER", None)
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def _load_research_tasks(jsonl_path: Path) -> list[dict]:
    tasks: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks


def _normalize_model_identifier(model_identifier: str) -> tuple[str, Path | None]:
    candidate = Path(model_identifier).expanduser()
    if candidate.exists():
        return str(candidate.resolve()), candidate.resolve()
    return model_identifier, None


def _apply_api_env(
    api_key: str | None,
    api_base_url: str | None,
    model_identifier: str,
) -> None:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["LLM_API_KEY"] = api_key
    if api_base_url:
        os.environ["OPENAI_BASE_URL"] = api_base_url
        os.environ["LLM_BASE_URL"] = api_base_url
    os.environ["LLM_MODEL"] = model_identifier


def _prepare_research_configs(
    model_identifier: str,
    output_dir: Path,
    profiles: Iterable[str],
    max_iterations: int,
) -> list[Path]:
    config_dir = output_dir / "runtime_configs" / "research"
    output_jsonl_dir = output_dir / "research_outputs"
    prepared_paths: list[Path] = []
    for profile_name in profiles:
        base_path = (
            MULTIAGENTBENCH_ROOT
            / "marble"
            / "configs"
            / "test_config_research"
            / profile_name
        )
        config = _read_yaml(base_path)
        output_jsonl_dir.mkdir(parents=True, exist_ok=True)
        config["llm"] = model_identifier
        config.setdefault("metrics", {})["evaluate_llm"] = model_identifier
        config.setdefault("environment", {})["max_iterations"] = max_iterations
        config.setdefault("engine_planner", {})["max_iterations"] = max_iterations
        config.setdefault("output", {})["file_path"] = str(
            output_jsonl_dir / f"{base_path.stem}_discussion_output.jsonl"
        )
        runtime_path = config_dir / base_path.name
        _write_yaml(runtime_path, config)
        prepared_paths.append(runtime_path)
    return prepared_paths


def _prepare_research_configs_from_jsonl(
    model_identifier: str,
    output_dir: Path,
    jsonl_path: Path,
    max_iterations: int,
    limit: int | None = None,
) -> list[Path]:
    config_dir = output_dir / "runtime_configs" / "research_full"
    output_jsonl_dir = output_dir / "research_outputs"
    output_jsonl_dir.mkdir(parents=True, exist_ok=True)
    tasks = _load_research_tasks(jsonl_path)
    if limit is not None:
        tasks = tasks[:limit]
    prepared_paths: list[Path] = []
    for payload in tasks:
        task_id = payload.get("task_id", len(prepared_paths) + 1)
        config = dict(payload)
        config["coordinate_mode"] = config.get("coordinate_mode") or "graph"
        config["llm"] = model_identifier
        config["environment"] = config.get("environment") or {}
        config["environment"]["max_iterations"] = max_iterations
        config["environment"]["name"] = config["environment"].get("name") or "Research Collaboration Environment"
        config["environment"]["type"] = config["environment"].get("type") or "Research"
        config["engine_planner"] = config.get("engine_planner") or {}
        config["engine_planner"]["max_iterations"] = max_iterations
        config["memory"] = config.get("memory") or {}
        config["memory"]["type"] = config["memory"].get("type") or "SharedMemory"
        config["metrics"] = config.get("metrics") or {}
        config["metrics"]["evaluate_llm"] = model_identifier
        config["output"] = config.get("output") or {}
        config["output"]["file_path"] = str(output_jsonl_dir / f"task_{task_id}_discussion_output.jsonl")
        config["output"]["format"] = config["output"].get("format") or "jsonl"
        runtime_path = config_dir / f"task_{task_id}.yaml"
        _write_yaml(runtime_path, config)
        prepared_paths.append(runtime_path)
    return prepared_paths


def _prepare_werewolf_config(model_identifier: str, output_dir: Path) -> Path:
    base_path = (
        MULTIAGENTBENCH_ROOT
        / "marble"
        / "configs"
        / "test_config"
        / "werewolf_config"
        / "werewolf_config.yaml"
    )
    config = _read_yaml(base_path)
    config["openai_api_key"] = "EMPTY"
    for key in ("villager_config", "werewolf_config"):
        section = config.setdefault(key, {})
        section["base_url"] = "http://localhost/v1"
        section["api_key"] = "EMPTY"
        section["model_name"] = model_identifier
    runtime_path = output_dir / "runtime_configs" / "werewolf_config.yaml"
    _write_yaml(runtime_path, config)
    return runtime_path


def _collect_new_game_dirs(before: set[Path], log_root: Path) -> list[Path]:
    after = {path for path in log_root.glob("game_*") if path.is_dir()}
    return sorted(after - before)


def _run_research_configs(
    python_bin: str,
    config_paths: list[Path],
    fast_mode: bool = True,
    in_process: bool = False,
) -> list[Path]:
    output_paths: list[Path] = []
    if in_process:
        sys.path.insert(0, str(MULTIAGENTBENCH_ROOT))
        from marble.configs.config import Config
        from marble.engine.engine import Engine

        with _pushd(MULTIAGENTBENCH_ROOT):
            if fast_mode:
                os.environ.setdefault("PREDICTDESIGN_FAST_TOOL_CALLS", "1")
                os.environ.setdefault("PREDICTDESIGN_LOCAL_MAX_NEW_TOKENS", "96")
                os.environ.setdefault("PREDICTDESIGN_FAST_LOCAL_PLANNER", "1")
            else:
                os.environ.pop("PREDICTDESIGN_FAST_TOOL_CALLS", None)
                os.environ.pop("PREDICTDESIGN_LOCAL_MAX_NEW_TOKENS", None)
                os.environ.pop("PREDICTDESIGN_FAST_LOCAL_PLANNER", None)
            for config_path in config_paths:
                print("$", python_bin, "marble/main.py", "--config_path", str(config_path))
                engine = Engine(Config.load(str(config_path)))
                engine.start()
                config = _read_yaml(config_path)
                output_path = Path(config["output"]["file_path"])
                if not output_path.exists():
                    raise FileNotFoundError(
                        f"Research run finished but output JSONL was not created: {output_path}"
                    )
                output_paths.append(output_path)
    else:
        for config_path in config_paths:
            _run_command(
                [python_bin, "marble/main.py", "--config_path", str(config_path)],
                cwd=MULTIAGENTBENCH_ROOT,
                fast_mode=fast_mode,
            )
            config = _read_yaml(config_path)
            output_path = Path(config["output"]["file_path"])
            if not output_path.exists():
                raise FileNotFoundError(
                    f"Research run finished but output JSONL was not created: {output_path}"
                )
            output_paths.append(output_path)
    return output_paths


def _run_werewolf_game(
    python_bin: str,
    config_path: Path,
    output_dir: Path,
    fast_mode: bool = True,
    games: int = 1,
    in_process: bool = False,
) -> list[Path]:
    log_root = MULTIAGENTBENCH_ROOT / "werewolf_log"
    log_root.mkdir(parents=True, exist_ok=True)
    before = {path for path in log_root.glob("game_*") if path.is_dir()}
    if in_process:
        sys.path.insert(0, str(MULTIAGENTBENCH_ROOT))
        from marble.environments.werewolf_env import start_game

        with _pushd(MULTIAGENTBENCH_ROOT):
            if fast_mode:
                os.environ.setdefault("PREDICTDESIGN_FAST_TOOL_CALLS", "1")
                os.environ.setdefault("PREDICTDESIGN_LOCAL_MAX_NEW_TOKENS", "96")
                os.environ.setdefault("PREDICTDESIGN_FAST_LOCAL_PLANNER", "1")
            else:
                os.environ.pop("PREDICTDESIGN_FAST_TOOL_CALLS", None)
                os.environ.pop("PREDICTDESIGN_LOCAL_MAX_NEW_TOKENS", None)
                os.environ.pop("PREDICTDESIGN_FAST_LOCAL_PLANNER", None)
            for _ in range(games):
                print("$", python_bin, "marble/environments/werewolf_env.py", "--rounds", "1", "--name", "predictdesign_local_qwen", "--config_path", str(config_path))
                start_game("predictdesign_local_qwen", str(config_path))
    else:
        _run_command(
            [
                python_bin,
                "marble/environments/werewolf_env.py",
                "--rounds",
                str(games),
                "--name",
                "predictdesign_local_qwen",
                "--config_path",
                str(config_path),
            ],
            cwd=MULTIAGENTBENCH_ROOT,
            fast_mode=fast_mode,
        )
    new_dirs = _collect_new_game_dirs(before, log_root)
    if not new_dirs:
        raise FileNotFoundError("Werewolf run did not create a new game_* log directory.")
    archived_root = output_dir / "werewolf_outputs"
    archived_root.mkdir(parents=True, exist_ok=True)
    archived_paths: list[Path] = []
    for path in new_dirs:
        target = archived_root / path.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(path, target)
        archived_paths.append(target)
    return archived_paths


def _evaluate(
    research_outputs: list[Path],
    werewolf_dirs: list[Path],
    report_path: Path,
    context_dim: int,
    hidden_dim: int,
    train_epochs: int,
    learning_rate: float,
    weight_decay: float,
    train_fraction: float,
    device: str,
    gnn_types: tuple[str, ...],
    llm_api_config: LLMApiConfig,
) -> list[dict]:
    adapter = MultiAgentBenchAdapter(
        context_dim=context_dim,
        hidden_dim=hidden_dim,
        device=device,
    )
    evaluator = BenchmarkEvaluator(
        context_dim=context_dim,
        hidden_dim=hidden_dim,
        device=device,
        train_epochs=train_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        train_fraction=train_fraction,
        llm_api_config=llm_api_config,
    )
    results = []
    if research_outputs:
        research_episodes = []
        for output_path in research_outputs:
            research_episodes.extend(adapter.load_research_from_output_jsonl(output_path))
        results.extend(evaluator.evaluate_dataset("research", research_episodes, gnn_types=gnn_types))
    if werewolf_dirs:
        werewolf_episodes = []
        for checkpoint_dir in werewolf_dirs:
            werewolf_episodes.extend(adapter.load_werewolf_from_checkpoints(checkpoint_dir))
        results.extend(evaluator.evaluate_dataset("werewolf", werewolf_episodes, gnn_types=gnn_types))
    evaluator.save_report(report_path, results)
    return [asdict(item) for item in results]


def _collect_existing_research_outputs(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        raise FileNotFoundError(f"Research output directory does not exist: {output_dir}")
    outputs = sorted(output_dir.glob("*_discussion_output.jsonl"))
    if not outputs:
        raise FileNotFoundError(f"No research output jsonl files found in: {output_dir}")
    return outputs


def _collect_existing_werewolf_dirs(output_dir: Path) -> list[Path]:
    if not output_dir.exists():
        raise FileNotFoundError(f"Werewolf output directory does not exist: {output_dir}")
    outputs = sorted(path for path in output_dir.glob("game_*") if path.is_dir())
    if not outputs:
        raise FileNotFoundError(f"No werewolf game_* directories found in: {output_dir}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MultiAgentBench research/werewolf with local Qwen and evaluate PredictDesign combinations."
    )
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--api-base-url", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--research-profiles", nargs="*", default=["profile_1.yaml"])
    parser.add_argument("--research-jsonl", type=str, default=None)
    parser.add_argument("--research-limit", type=int, default=None)
    parser.add_argument("--research-max-iterations", type=int, default=3)
    parser.add_argument("--werewolf-games", type=int, default=1)
    parser.add_argument("--existing-research-output-dir", type=str, default=None)
    parser.add_argument("--existing-werewolf-output-dir", type=str, default=None)
    parser.add_argument("--skip-research-run", action="store_true")
    parser.add_argument("--skip-werewolf-run", action="store_true")
    parser.add_argument("--full-fidelity", action="store_true")
    parser.add_argument("--in-process", action="store_true")
    parser.add_argument("--context-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--train-epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--gnn-types", nargs="*", default=["gcn", "graphsage", "gat"])
    parser.add_argument("--llm-api-key", type=str, default=LLMApiConfig().api_key)
    parser.add_argument("--llm-base-url", type=str, default=LLMApiConfig().base_url)
    parser.add_argument("--llm-model", type=str, default=LLMApiConfig().model)
    parser.add_argument("--llm-temperature", type=float, default=LLMApiConfig().temperature)
    parser.add_argument("--llm-max-tokens", type=int, default=LLMApiConfig().max_tokens)
    parser.add_argument("--llm-timeout", type=float, default=LLMApiConfig().timeout)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    model_identifier, resolved_model_path = _normalize_model_identifier(args.model_path)
    if resolved_model_path is None and not model_identifier.strip():
        raise ValueError("model identifier must not be empty.")
    _apply_api_env(
        api_key=args.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY"),
        api_base_url=args.api_base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_BASE_URL"),
        model_identifier=model_identifier,
    )

    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.research_jsonl:
        research_config_paths = _prepare_research_configs_from_jsonl(
            model_identifier=model_identifier,
            output_dir=results_dir,
            jsonl_path=Path(args.research_jsonl).resolve(),
            max_iterations=args.research_max_iterations,
            limit=args.research_limit,
        )
    else:
        research_config_paths = _prepare_research_configs(
            model_identifier=model_identifier,
            output_dir=results_dir,
            profiles=args.research_profiles,
            max_iterations=args.research_max_iterations,
        )
    werewolf_config_path = _prepare_werewolf_config(
        model_identifier=model_identifier,
        output_dir=results_dir,
    )

    research_outputs: list[Path] = []
    werewolf_dirs: list[Path] = []
    if not args.skip_research_run:
        research_outputs = _run_research_configs(
            args.python_bin,
            research_config_paths,
            fast_mode=not args.full_fidelity,
            in_process=args.in_process,
        )
    elif args.existing_research_output_dir:
        research_outputs = _collect_existing_research_outputs(
            Path(args.existing_research_output_dir).resolve()
        )
    if not args.skip_werewolf_run:
        werewolf_dirs = _run_werewolf_game(
            args.python_bin,
            werewolf_config_path,
            results_dir,
            fast_mode=not args.full_fidelity,
            games=args.werewolf_games,
            in_process=args.in_process,
        )
    elif args.existing_werewolf_output_dir:
        werewolf_dirs = _collect_existing_werewolf_dirs(
            Path(args.existing_werewolf_output_dir).resolve()
        )

    report_path = results_dir / "multiagentbench_accuracy_report.json"
    serializable_results = _evaluate(
        research_outputs=research_outputs,
        werewolf_dirs=werewolf_dirs,
        report_path=report_path,
        context_dim=args.context_dim,
        hidden_dim=args.hidden_dim,
        train_epochs=args.train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_fraction=args.train_fraction,
        device=args.device,
        gnn_types=tuple(args.gnn_types),
        llm_api_config=LLMApiConfig(
            api_key=args.llm_api_key,
            base_url=args.llm_base_url,
            model=args.llm_model,
            temperature=args.llm_temperature,
            max_tokens=args.llm_max_tokens,
            timeout=args.llm_timeout,
        ),
    )
    print(report_path)
    print(json.dumps(serializable_results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
