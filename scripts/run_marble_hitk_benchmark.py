from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TextIO

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predictdesign.benchmark.evaluator import BenchmarkEvaluator
from predictdesign.benchmark.multiagentbench import MultiAgentBenchAdapter
from predictdesign.benchmark.rich_log import train_mlp_on_rich_log, write_rich_log
from predictdesign.config import LLMApiConfig


MARBLE_ROOT = PROJECT_ROOT / "prefetch-kv-mas" / "benchmarks" / "marble"
CODING_RENDER_SCRIPT = MARBLE_ROOT / "scripts" / "coding" / "update_coding_config.py"
RESEARCH_RENDER_SCRIPT = MARBLE_ROOT / "scripts" / "research" / "jsonl2yaml.py"
CODING_TEMPLATE = MARBLE_ROOT / "core" / "configs" / "coding_config" / "coding_config.yaml"
RESEARCH_JSONL = MARBLE_ROOT / "metadata" / "research" / "research_main.jsonl"
CODING_JSONL = MARBLE_ROOT / "metadata" / "coding" / "coding_main.jsonl"
WEREWOLF_TEMPLATE = (
    MARBLE_ROOT / "core" / "configs" / "werewolf_config" / "werewolf_config.yaml"
)

DEFAULT_TASK_IDS = (7, 17, 27, 37, 47, 57, 67, 77, 87, 97)


def _default_python_bin() -> str:
    windows_venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    posix_venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    for candidate in (windows_venv_python, posix_venv_python):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _python_bin() -> str:
    return os.environ.get("PREDICTDESIGN_PYTHON_BIN") or _default_python_bin()


def _default_results_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "results" / "marble_hitk_runs" / timestamp


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _file_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(message)
        if not message.endswith("\n"):
            handle.write("\n")
        handle.flush()


def _console(message: str) -> None:
    print(message, flush=True)


def _write_log_line(handle: TextIO, line: str) -> None:
    handle.write(line)
    if not line.endswith("\n"):
        handle.write("\n")
    handle.flush()


def _tee_subprocess_output(env: dict[str, str]) -> bool:
    value = env.get("PREDICTDESIGN_TEE_SUBPROCESS", os.environ.get("PREDICTDESIGN_TEE_SUBPROCESS", "0"))
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _run(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path | None = None) -> None:
    command_text = "$ " + " ".join(command)
    if log_path is not None:
        _console(f"{command_text}\n  log: {log_path}")
    else:
        _console(command_text)
    if log_path is None:
        subprocess.run(command, cwd=str(cwd), env=env, check=True)
        return
    tee_output = _tee_subprocess_output(env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="replace", newline="\n") as handle:
        _write_log_line(handle, f"\n===== {_timestamp()} COMMAND START =====")
        _write_log_line(handle, command_text)
        _write_log_line(handle, f"cwd={cwd}")
        if tee_output:
            _console(f"[{_timestamp()}] streaming subprocess output -> {log_path}")
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            _write_log_line(handle, line)
            if tee_output:
                print(line, end="", flush=True)
        return_code = process.wait()
        _write_log_line(handle, f"===== {_timestamp()} COMMAND END returncode={return_code} =====")
        if tee_output:
            _console(f"[{_timestamp()}] command finished returncode={return_code}")
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _resume_enabled() -> bool:
    return os.environ.get("PREDICTDESIGN_RESUME", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }


def _completed_file(path: Path) -> bool:
    if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
        return False
    if path.suffix.lower() != ".jsonl":
        return True
    has_record = False
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                json.loads(stripped)
                has_record = True
    except json.JSONDecodeError:
        return False
    return has_record


def _archive_incomplete_path(path: Path, log_path: Path, *, reason: str) -> Path | None:
    if not path.exists():
        return None
    target = path.with_name(f"{path.name}.incomplete_{_file_timestamp()}")
    index = 1
    while target.exists():
        target = path.with_name(f"{path.name}.incomplete_{_file_timestamp()}_{index}")
        index += 1
    shutil.move(str(path), str(target))
    message = f"[resume] archived incomplete {path} -> {target}; reason={reason}"
    _console(message)
    _append_log(log_path, f"[{_timestamp()}] {message}")
    return target


def _completed_werewolf_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return len(list(path.glob("checkpoint_*.json"))) >= 2


def _base_env(config: LLMApiConfig) -> dict[str, str]:
    env = dict(os.environ)
    env["OPENAI_API_KEY"] = config.api_key
    env["OPENAI_API_BASE"] = config.base_url
    env["OPENAI_BASE_URL"] = config.base_url
    env["LLM_API_KEY"] = config.api_key
    env["LLM_API_BASE"] = config.base_url
    env["LLM_BASE_URL"] = config.base_url
    env["LLM_MODEL"] = config.model
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


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


def _prepare_coding_config(
    task_id: int,
    config: LLMApiConfig,
    runtime_dir: Path,
    output_dir: Path,
    env: dict[str, str],
    log_path: Path | None = None,
) -> tuple[Path, Path]:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    config_path = runtime_dir / f"task_{task_id}.yaml"
    workspace_dir = runtime_dir / f"task_{task_id}_workspace"
    output_path = output_dir / f"task_{task_id}_result.jsonl"
    _run(
        [
            _python_bin(),
            str(CODING_RENDER_SCRIPT),
            "--benchmark_id",
            str(task_id),
            "--model_name",
            config.model,
            "--workspace_dir",
            str(workspace_dir),
            "--config_path",
            str(CODING_TEMPLATE),
            "--benchmark_path",
            str(CODING_JSONL),
            "--output_path",
            str(config_path),
        ],
        cwd=MARBLE_ROOT,
        env=env,
        log_path=log_path,
    )
    rendered = _read_yaml(config_path)
    rendered.setdefault("output", {})["file_path"] = str(output_path)
    rendered.setdefault("output", {})["format"] = "jsonl"
    rendered["llm"] = config.model
    rendered.setdefault("metrics", {})["evaluate_llm"] = config.model
    _write_yaml(config_path, rendered)
    return config_path, output_path


def _prepare_research_config(
    task_id: int,
    config: LLMApiConfig,
    runtime_dir: Path,
    output_dir: Path,
    env: dict[str, str],
    log_path: Path | None = None,
) -> tuple[Path, Path]:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    config_path = runtime_dir / f"task_{task_id}.yaml"
    output_path = output_dir / f"research_{task_id}_result.jsonl"
    default_environment = json.dumps(
        {
            "max_iterations": 3,
            "name": "Research Collaboration Environment",
            "type": "Research",
        }
    )
    default_output = json.dumps({"file_path": str(output_path), "format": "jsonl"})
    _run(
        [
            _python_bin(),
            str(RESEARCH_RENDER_SCRIPT),
            "--input_file",
            str(RESEARCH_JSONL),
            "--task_id",
            str(task_id),
            "--output_file",
            str(config_path),
            "--default_llm",
            config.model,
            "--default_metrics_evaluate_llm",
            config.model,
            "--default_environment",
            default_environment,
            "--default_output",
            default_output,
        ],
        cwd=MARBLE_ROOT,
        env=env,
        log_path=log_path,
    )
    return config_path, output_path


def _prepare_werewolf_config(
    config: LLMApiConfig,
    runtime_dir: Path,
    log_path: Path | None = None,
) -> Path:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    config_path = runtime_dir / "werewolf_config.yaml"
    rendered = _read_yaml(WEREWOLF_TEMPLATE)
    rendered["openai_api_key"] = config.api_key
    for key in ("villager_config", "werewolf_config"):
        section = rendered.setdefault(key, {})
        section["base_url"] = config.base_url
        section["api_key"] = config.api_key
        section["model_name"] = config.model
    _write_yaml(config_path, rendered)
    if log_path is not None:
        _append_log(log_path, f"[{_timestamp()}] wrote werewolf config: {config_path}")
    return config_path


def _new_werewolf_dirs(before: set[Path], log_root: Path) -> list[Path]:
    after = {path for path in log_root.glob("game_*") if path.is_dir()}
    return sorted(after - before)


def _run_coding(
    task_ids: tuple[int, ...],
    config: LLMApiConfig,
    results_dir: Path,
    env: dict[str, str],
) -> list[Path]:
    runtime_dir = results_dir / "runtime_configs" / "coding"
    output_dir = results_dir / "coding_outputs"
    log_dir = results_dir / "logs" / "coding"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for task_id in task_ids:
        output_path = output_dir / f"task_{task_id}_result.jsonl"
        log_path = log_dir / f"task_{task_id}.log"
        _console(f"[coding] task={task_id} start log={log_path}")
        _append_log(log_path, f"[{_timestamp()}] coding task {task_id} start resume={_resume_enabled()}")
        if _resume_enabled() and _completed_file(output_path):
            message = f"[resume] coding task {task_id} already completed: {output_path}"
            _console(message)
            _append_log(log_path, f"[{_timestamp()}] {message}")
            outputs.append(output_path)
            continue
        if _resume_enabled():
            _archive_incomplete_path(
                runtime_dir / f"task_{task_id}_workspace",
                log_path,
                reason="coding output is missing, empty, or invalid JSONL",
            )
            if output_path.exists():
                _archive_incomplete_path(
                    output_path,
                    log_path,
                    reason="coding output is incomplete and will be regenerated",
                )
        config_path, output_path = _prepare_coding_config(
            task_id,
            config,
            runtime_dir,
            output_dir,
            env,
            log_path=log_path,
        )
        _run(
            [_python_bin(), "-m", "core.main", "--config_path", str(config_path)],
            cwd=MARBLE_ROOT,
            env=env,
            log_path=log_path,
        )
        if not output_path.exists():
            raise FileNotFoundError(f"Coding output missing: {output_path}")
        _append_log(log_path, f"[{_timestamp()}] coding task {task_id} done output={output_path}")
        _console(f"[coding] task={task_id} done output={output_path}")
        outputs.append(output_path)
    return outputs


def _run_research(
    task_ids: tuple[int, ...],
    config: LLMApiConfig,
    results_dir: Path,
    env: dict[str, str],
) -> list[Path]:
    runtime_dir = results_dir / "runtime_configs" / "research"
    output_dir = results_dir / "research_outputs"
    log_dir = results_dir / "logs" / "research"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for task_id in task_ids:
        output_path = output_dir / f"research_{task_id}_result.jsonl"
        log_path = log_dir / f"task_{task_id}.log"
        _console(f"[research] task={task_id} start log={log_path}")
        _append_log(log_path, f"[{_timestamp()}] research task {task_id} start resume={_resume_enabled()}")
        if _resume_enabled() and _completed_file(output_path):
            message = f"[resume] research task {task_id} already completed: {output_path}"
            _console(message)
            _append_log(log_path, f"[{_timestamp()}] {message}")
            outputs.append(output_path)
            continue
        if _resume_enabled() and output_path.exists():
            _archive_incomplete_path(
                output_path,
                log_path,
                reason="research output is empty or invalid JSONL and will be regenerated",
            )
        config_path, output_path = _prepare_research_config(
            task_id,
            config,
            runtime_dir,
            output_dir,
            env,
            log_path=log_path,
        )
        _run(
            [_python_bin(), "-m", "core.main", "--config_path", str(config_path)],
            cwd=MARBLE_ROOT,
            env=env,
            log_path=log_path,
        )
        if not output_path.exists():
            raise FileNotFoundError(f"Research output missing: {output_path}")
        _append_log(log_path, f"[{_timestamp()}] research task {task_id} done output={output_path}")
        _console(f"[research] task={task_id} done output={output_path}")
        outputs.append(output_path)
    return outputs


def _run_werewolf(
    games: int,
    config: LLMApiConfig,
    results_dir: Path,
    env: dict[str, str],
) -> list[Path]:
    runtime_dir = results_dir / "runtime_configs" / "werewolf"
    archived_root = results_dir / "werewolf_outputs"
    log_dir = results_dir / "logs" / "werewolf"
    archived_root.mkdir(parents=True, exist_ok=True)
    config_path = _prepare_werewolf_config(config, runtime_dir, log_path=log_dir / "config.log")
    log_root = MARBLE_ROOT / "results" / "werewolf"
    log_root.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    if _resume_enabled():
        completed = [
            path
            for path in sorted(archived_root.glob("game_*"))
            if _completed_werewolf_dir(path)
        ]
        outputs.extend(completed[:games])
        for path in outputs:
            message = f"[resume] werewolf game already completed: {path}"
            _console(message)
            _append_log(log_dir / "resume.log", f"[{_timestamp()}] {message}")
    for game_index in range(len(outputs) + 1, games + 1):
        log_path = log_dir / f"game_{game_index}.log"
        _console(f"[werewolf] game={game_index} start log={log_path}")
        _append_log(log_path, f"[{_timestamp()}] werewolf game {game_index} start resume={_resume_enabled()}")
        before = {path for path in log_root.glob("game_*") if path.is_dir()}
        _run(
            [
                _python_bin(),
                "-m",
                "core.environments.werewolf_env",
                "--rounds",
                "1",
                "--name",
                f"predictdesign_werewolf_{game_index}",
                "--config_path",
                str(config_path),
            ],
            cwd=MARBLE_ROOT,
            env=env,
            log_path=log_path,
        )
        new_dirs = _new_werewolf_dirs(before, log_root)
        if not new_dirs:
            raise FileNotFoundError("Werewolf run did not create a new game_* directory.")
        for path in new_dirs:
            target = archived_root / path.name
            if target.exists():
                _archive_incomplete_path(
                    target,
                    log_path,
                    reason="werewolf archive exists but was not marked complete",
                )
            shutil.copytree(path, target)
            _append_log(log_path, f"[{_timestamp()}] archived werewolf game {path} -> {target}")
            _console(f"[werewolf] game={game_index} archived output={target}")
            outputs.append(target)
    return outputs


def _evaluate(
    coding_outputs: list[Path],
    research_outputs: list[Path],
    werewolf_outputs: list[Path],
    config: LLMApiConfig,
    device: str,
    results_dir: Path,
    rich_log_path: Path | None = None,
    train_rich_log_mlp: bool = False,
    mlp_output_dir: Path | None = None,
    mlp_max_samples: int = 100,
    mlp_epochs: int = 60,
    mlp_label_mode: str = "action_type",
) -> dict[str, dict[str, float]]:
    adapter = MultiAgentBenchAdapter(context_dim=16, hidden_dim=32, device=device)
    evaluator = BenchmarkEvaluator(
        context_dim=16,
        hidden_dim=32,
        device=device,
        train_epochs=0,
        hit_k_values=(1, 3, 5),
        llm_api_config=config,
    )
    summary: dict[str, dict[str, float]] = {}
    report_rows = []
    all_episodes = []

    if coding_outputs:
        episodes = []
        for path in coding_outputs:
            episodes.extend(adapter.load_coding_from_output_jsonl(path))
        all_episodes.extend(episodes)
        result = evaluator.evaluate_dataset("coding", episodes, gnn_types=("llm_api",))[0]
        summary["coding"] = {f"hit@{k}": float(result.hit_at_k[str(k)]) for k in result.hit_ks}
        report_rows.append(asdict(result))

    if research_outputs:
        episodes = []
        for path in research_outputs:
            episodes.extend(adapter.load_research_from_output_jsonl(path))
        all_episodes.extend(episodes)
        result = evaluator.evaluate_dataset("research", episodes, gnn_types=("llm_api",))[0]
        summary["research"] = {f"hit@{k}": float(result.hit_at_k[str(k)]) for k in result.hit_ks}
        report_rows.append(asdict(result))

    if werewolf_outputs:
        episodes = []
        for path in werewolf_outputs:
            episodes.extend(adapter.load_werewolf_from_checkpoints(path))
        all_episodes.extend(episodes)
        result = evaluator.evaluate_dataset("werewolf", episodes, gnn_types=("llm_api",))[0]
        summary["werewolf"] = {f"hit@{k}": float(result.hit_at_k[str(k)]) for k in result.hit_ks}
        report_rows.append(asdict(result))

    report_path = results_dir / "hitk_report.json"
    report_path.write_text(json.dumps(report_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path = results_dir / "hitk_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if rich_log_path is not None:
        rich_result = write_rich_log(
            rich_log_path,
            all_episodes,
            context_dim=16,
            device="cpu",
        )
        print(
            f"rich_log={rich_result.path} records={rich_result.record_count} "
            f"episodes={rich_result.episode_count}"
        )
        if train_rich_log_mlp:
            mlp_result = train_mlp_on_rich_log(
                log_path=rich_result.path,
                output_dir=mlp_output_dir or (results_dir / "rich_log_mlp"),
                max_samples=mlp_max_samples,
                epochs=mlp_epochs,
                label_mode=mlp_label_mode,
            )
            print(f"mlp_report={mlp_result.report_path}")
            print(f"mlp_chart={mlp_result.chart_path}")
    return summary


def _parse_task_ids(values: list[int] | None) -> tuple[int, ...]:
    if not values:
        return DEFAULT_TASK_IDS
    return tuple(dict.fromkeys(int(value) for value in values))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run marble coding/research/werewolf and evaluate PredictDesign hit@k."
    )
    parser.add_argument("--coding-task-ids", nargs="*", type=int, default=list(DEFAULT_TASK_IDS))
    parser.add_argument("--research-task-ids", nargs="*", type=int, default=list(DEFAULT_TASK_IDS))
    parser.add_argument("--werewolf-games", type=int, default=10)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--results-dir", type=str, default=str(_default_results_dir()))
    parser.add_argument("--rich-log-path", type=str, default=None)
    parser.add_argument("--train-rich-log-mlp", action="store_true")
    parser.add_argument("--mlp-output-dir", type=str, default=None)
    parser.add_argument("--mlp-max-samples", type=int, default=100)
    parser.add_argument("--mlp-epochs", type=int, default=60)
    parser.add_argument(
        "--mlp-label-mode",
        choices=("action_type", "action_signature"),
        default="action_type",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    config = LLMApiConfig()
    os.environ["PREDICTDESIGN_RESUME"] = "1" if args.resume else "0"
    env = _base_env(config)
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    coding_outputs = _run_coding(_parse_task_ids(args.coding_task_ids), config, results_dir, env)
    research_outputs = _run_research(_parse_task_ids(args.research_task_ids), config, results_dir, env)
    werewolf_outputs = _run_werewolf(args.werewolf_games, config, results_dir, env)
    _evaluate(
        coding_outputs=coding_outputs,
        research_outputs=research_outputs,
        werewolf_outputs=werewolf_outputs,
        config=config,
        device=args.device,
        results_dir=results_dir,
        rich_log_path=(
            Path(args.rich_log_path).resolve()
            if args.rich_log_path
            else (results_dir / "rich_log.jsonl" if args.train_rich_log_mlp else None)
        ),
        train_rich_log_mlp=args.train_rich_log_mlp,
        mlp_output_dir=Path(args.mlp_output_dir).resolve() if args.mlp_output_dir else None,
        mlp_max_samples=args.mlp_max_samples,
        mlp_epochs=args.mlp_epochs,
        mlp_label_mode=args.mlp_label_mode,
    )


if __name__ == "__main__":
    main()
