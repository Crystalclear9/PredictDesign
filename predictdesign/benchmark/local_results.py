from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .multiagentbench import MultiAgentBenchAdapter
from .trainer import BenchmarkSplit, BenchmarkTrainer
from .types import BenchmarkEpisode


@dataclass(slots=True)
class DatasetCorpus:
    dataset_name: str
    source_paths: list[Path]
    episodes: list[BenchmarkEpisode]

    @property
    def source_count(self) -> int:
        return len(self.source_paths)

    @property
    def episode_count(self) -> int:
        return len(self.episodes)


@dataclass(slots=True)
class ParallelApiCorpus:
    root_path: Path
    datasets: dict[str, DatasetCorpus]

    def dataset_splits(self, trainer: BenchmarkTrainer) -> dict[str, BenchmarkSplit]:
        return {
            dataset_name: trainer.split_episodes(corpus.episodes)
            for dataset_name, corpus in self.datasets.items()
        }

    def combined_split(self, trainer: BenchmarkTrainer) -> BenchmarkSplit:
        train_episodes: list[BenchmarkEpisode] = []
        eval_episodes: list[BenchmarkEpisode] = []
        for split in self.dataset_splits(trainer).values():
            train_episodes.extend(split.train_episodes)
            eval_episodes.extend(split.eval_episodes)
        return BenchmarkSplit(train_episodes=train_episodes, eval_episodes=eval_episodes)


def load_parallel_api_corpus(
    root_path: str | Path,
    adapter: MultiAgentBenchAdapter,
) -> ParallelApiCorpus:
    root = Path(root_path).resolve()
    coding_dir = root / "coding_outputs"
    research_dir = root / "research_outputs"
    werewolf_dir = root / "werewolf_outputs"
    datasets = {
        "coding": DatasetCorpus(
            dataset_name="coding",
            source_paths=sorted(coding_dir.glob("*.jsonl")),
            episodes=_load_coding_episodes(adapter, coding_dir),
        ),
        "research": DatasetCorpus(
            dataset_name="research",
            source_paths=sorted(research_dir.glob("*.jsonl")),
            episodes=_load_research_episodes(adapter, research_dir),
        ),
        "werewolf": DatasetCorpus(
            dataset_name="werewolf",
            source_paths=(
                sorted(path for path in werewolf_dir.iterdir() if path.is_dir())
                if werewolf_dir.exists()
                else []
            ),
            episodes=_load_werewolf_episodes(adapter, werewolf_dir),
        ),
    }
    missing = [
        dataset_name
        for dataset_name, corpus in datasets.items()
        if corpus.source_count == 0 or corpus.episode_count == 0
    ]
    if missing:
        joined = ", ".join(sorted(missing))
        raise FileNotFoundError(
            f"Parallel API results are incomplete under {root}. Missing usable episodes for: {joined}."
        )
    return ParallelApiCorpus(root_path=root, datasets=datasets)


def _load_coding_episodes(
    adapter: MultiAgentBenchAdapter,
    coding_dir: Path,
) -> list[BenchmarkEpisode]:
    if not coding_dir.exists():
        return []
    episodes: list[BenchmarkEpisode] = []
    for output_path in sorted(coding_dir.glob("*.jsonl")):
        episodes.extend(adapter.load_coding_from_output_jsonl(output_path))
    return episodes


def _load_research_episodes(
    adapter: MultiAgentBenchAdapter,
    research_dir: Path,
) -> list[BenchmarkEpisode]:
    if not research_dir.exists():
        return []
    episodes: list[BenchmarkEpisode] = []
    for output_path in sorted(research_dir.glob("*.jsonl")):
        episodes.extend(adapter.load_research_from_output_jsonl(output_path))
    return episodes


def _load_werewolf_episodes(
    adapter: MultiAgentBenchAdapter,
    werewolf_dir: Path,
) -> list[BenchmarkEpisode]:
    if not werewolf_dir.exists():
        return []
    episodes: list[BenchmarkEpisode] = []
    for checkpoint_dir in sorted(path for path in werewolf_dir.iterdir() if path.is_dir()):
        episodes.extend(adapter.load_werewolf_from_checkpoints(checkpoint_dir))
    return episodes
