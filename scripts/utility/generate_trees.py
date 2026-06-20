"""
My test script for dialog tree generation.
Keeps the frontiers of all trees and expands.

We can use asyncio with vLLM to keep the logic for each tree separate and add queries to the queue for each model
independently. This greatly simplifies the logic for managing the frontiers of all trees and batching input.
"""

import asyncio
import traceback
import random
import uuid
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager
from codetiming import Timer
from rich.tree import Tree
from rich import print
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

from clarification_trees_v3.models import BidirectionalEntailmentClusterer, construct_semantic_clusterer
from clarification_trees_v3.utils import add_inference_messages, set_seed, add_cq_messages, add_answer_messages, SentenceAnalyzer
from clarification_trees_v3.dataset.dataset import ClearVQASample, ClearVQADataset
from clarification_trees_v3.dataset.dialog_tree import DialogTree, NodeType, DialogTrajectory, TreeSidecar
from clarification_trees_v3.models.remote_vllm_model import RemoteVLLMModel
from clarification_trees_v3.config.schema import parse_config, Config
from clarification_trees_v3.definitions import GENERATED_TREES_PATH, BASE_WEIGHTS_PATH

from clarification_trees_v3.models.utils import use_models
from clarification_trees_v3.dataset.tree_generation import process_dataset_lazily, print_timer_tree, expand_tree

async def run_single_tree_test(cfg: Config, raw_cfg: DictConfig, sample: ClearVQASample, base_dir: Path):
    test_tree = DialogTree(
        init_question=sample.blurred_question,
        init_image=None,
        init_image_path=sample.image_path,
        init_image_caption=sample.caption,
        unambiguous_question=sample.question,
        gold_answer=sample.gold_answer,
        answers=sample.answers
    )

    sentence_analyzer = SentenceAnalyzer()

    async with use_models(cfg, raw_cfg) as (cq_model, answer_model, clusterer):
        out_dir = base_dir / "single_tree_test"
        out_dir.mkdir(parents=True, exist_ok=True)
        with Timer("total"):
            await expand_tree(
                cfg=cfg,
                tree=test_tree,
                clusterer=clusterer,
                cq_model=cq_model,
                answer_model=answer_model,
                sentence_analyzer=sentence_analyzer,
                out_dir=out_dir
            )

    print_timer_tree()

async def run_expand_trees(cfg: Config, raw_cfg: DictConfig, ds: ClearVQADataset, out_dir: Path, n_parallel_trees: int = 10):
    sentence_analyzer = SentenceAnalyzer()

    async with use_models(cfg, raw_cfg) as (cq_model, answer_model, clusterer):
        with Timer("total"):
            await process_dataset_lazily(
                cfg=cfg,
                dataset=ds,
                indices=None,
                clusterer=clusterer,
                cq_model=cq_model,
                answer_model=answer_model,
                sentence_analyzer=sentence_analyzer,
                N_parallel_trees=n_parallel_trees,
                out_dir=out_dir
            )

    print_timer_tree()

@hydra.main(config_path="../../config", config_name="generate_trees", version_base=None)
def main(raw_cfg: DictConfig):
    cfg = parse_config(raw_cfg)
    SINGLE_TREE_TEST = False
    print(raw_cfg)
    set_seed(cfg.seed)

    if GENERATED_TREES_PATH is None:
        raise ValueError("GENERATED_TREES_PATH environment variable is required")

    out_path = GENERATED_TREES_PATH / cfg.paths.data.trees_subpath
    out_path.mkdir(parents=True, exist_ok=True)

    ds = ClearVQADataset(load_images=False, table_name="train_annotated.jsonl")

    if SINGLE_TREE_TEST:
        sample = ds[0]
        asyncio.run(run_single_tree_test(cfg, raw_cfg, sample, out_path))
    else:
        N_parallel_trees = 25

        # from torch.utils.data import Subset
        # ds = Subset(ds, range(25))

        asyncio.run(run_expand_trees(cfg, raw_cfg, ds, out_path, n_parallel_trees=N_parallel_trees))


if __name__ == "__main__":
    main()
