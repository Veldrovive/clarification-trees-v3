from clarification_trees_v3.dataset.dialog_tree import NodeType
from omegaconf import DictConfig
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
from typing import TypedDict

from clarification_trees_v3.dataset.dialog_tree import DialogTree, TreeSidecar, DialogTrajectory
import clarification_trees_v3.config.schema as schema

from clarification_trees_v3.definitions import GENERATED_TREES_PATH, CLEAR_VQA_BASE_PATH

@dataclass
class ClearVQASample:
    question_id: str
    question: str
    gold_answer: str
    image: Image.Image | None
    answers: list[str]
    dataset: str
    caption: str
    blurred_question: str
    clarification_question: str
    prompt_type: str
    image_path: Path
    ambiguity_category: str | None = None
    

class ClearVQADataset(Dataset):
    def __init__(self,
        data_path: Path = CLEAR_VQA_BASE_PATH,
        table_name = "train_annotated.jsonl",
        transform = None,
        load_images: bool = True
    ):
        self.transform = transform
        self.load_images = load_images

        self.data_path = data_path
        self.images_path = self.data_path / "images"
        assert self.images_path.is_dir(), "Images directory does not exist"
        self.table_path = self.data_path / table_name
        assert self.table_path.exists(), f"Table file {self.table_path} does not exist"
        self.load_tables()

        print(f"Train headers: {self.table_df.columns}")

    def load_tables(self):
        print("Loading table...")
        self.table_df = pd.read_json(self.table_path, lines=True)
        print("Finished loading tables")

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, index) -> ClearVQASample:
        sample = self.table_df.iloc[index].to_dict()

        # The loaded sample has the image as a string (the name of the image)
        # We use this to compute the image path and replace it with the actual image object if we are loading images
        image_name = sample['image']
        image_path = self.images_path / image_name
        assert image_path.exists(), f"Image not found at {image_path}"
        sample['image_path'] = image_path
        sample['image'] = None
        
        if self.load_images:
            if image_path.exists():
                try:
                    img = Image.open(image_path).convert('RGB')
                    if self.transform:
                        img = self.transform(img)
                    sample['image'] = img
                except FileNotFoundError:
                    raise FileNotFoundError(f"Image not found at {image_path}")
            else:
                raise FileNotFoundError(f"Image not found at {image_path}")
        
        return ClearVQASample(**sample)


@dataclass
class ClarificationTreeSample:
    tree: DialogTree
    tree_sidecar: TreeSidecar
    parent_node_idx: int
    child_node_idxs: list[int]
    advantages: list[float]
    token_values: list[torch.Tensor] | None
    token_logprobs: list[torch.Tensor] | None

class ClarificationTreeSampleDict(TypedDict):
    tree_idx: int
    parent_node_idx: int
    child_node_idxs: list[int]

class ClarificationTreeDataset(Dataset):
    trees: list[DialogTree]
    sidecars: list[TreeSidecar]
    samples: list[ClarificationTreeSampleDict]
    cached_reward_tree_idxs: set[int]
    
    def __init__(self,
        cfg: schema.Config | None,
        trees_path: Path | None = GENERATED_TREES_PATH,
        tree_paths: list[Path] | None = None,
        transform = None,
        load_images: bool = True,
        precompute_rewards: bool = True,
    ):
        self.cfg = cfg
        self.trees_path = trees_path
        self.tree_paths = tree_paths
        self.transform = transform
        self.load_images = load_images
        self.precompute_rewards = precompute_rewards

        assert self.trees_path is None or self.tree_paths is None, "One of trees_path or tree_paths must be None"
        assert self.tree_paths is not None or self.tree_paths is not None, "One of trees_path or tree_paths must not be None"

        self._load_trees()

    def _load_trees(self):
        trees = []
        sidecars = []
        cached_reward_tree_idxs = set()

        samples = []  # [{"tree_idx": int, "parent_node_idx": int, "child_node_idxs": list[int]}]

        if self.trees_path is not None:
            tree_dirs = list(self.trees_path.iterdir())
        elif self.tree_paths is not None:
            tree_dirs = self.tree_paths
        else:
            raise ValueError(f"Both trees_path and tree_path")

        num_parents = 0
        num_children = 0
        num_removed_children = 0
        for tree_dir in tqdm(tree_dirs, desc="Loading trees"):
            if not tree_dir.is_dir():
                print(f"Skipping non-directory {tree_dir}")
                continue

            tree_path = tree_dir / "tree.json"
            if not tree_path.exists():
                print(f"Skipping non-tree {tree_path}")
                continue
            
            sidecar_path = tree_dir / "tree_sidecar.json"
            if not sidecar_path.exists():
                print(f"Skipping non-sidecar {sidecar_path}")
                continue
            
            tree = DialogTree.load(tree_path, load_images=self.load_images)
            sidecar = TreeSidecar.load(sidecar_path)
            
            tree_idx = len(trees)
            trees.append(tree)
            sidecars.append(sidecar)
            if self.precompute_rewards:
                assert self.cfg is not None, "Config must be provided to compute rewards"
                sidecar.compute_rewards(self.cfg)  # Caches the rewards and advantages
                cached_reward_tree_idxs.add(tree_idx)

            for parent_node_idx, parent_node in tree.get_nodes():
                child_cq_idxs = tree.get_children_idxs(parent_node_idx, type_filter=NodeType.CLARIFICATION_QUESTION)

                # Append only the children that have a non-zero advantage
                unfiltered_child_size = len(child_cq_idxs)
                if unfiltered_child_size == 0:
                    continue

                child_cq_idxs = [child_node_idx for child_node_idx in child_cq_idxs if sidecar.advantage_cache[child_node_idx] != 0]
                filtered_child_size = len(child_cq_idxs)

                num_removed_children += unfiltered_child_size - filtered_child_size
                num_children += filtered_child_size

                if filtered_child_size == 0:
                    continue
            
                num_parents += 1
                
                samples.append(ClarificationTreeSampleDict(
                    tree_idx=tree_idx,
                    parent_node_idx=parent_node_idx,
                    child_node_idxs=child_cq_idxs
                ))

        self.trees: list[DialogTree] = trees
        self.sidecars: list[TreeSidecar] = sidecars
        self.samples: list[ClarificationTreeSampleDict] = samples
        self.cached_reward_tree_idxs: set[int] = cached_reward_tree_idxs

        print(f"Found {len(self.trees)} trees for a total of {len(self.samples)} samples.")
        print(f"Ended with {num_parents} parents, {num_children} children, and {num_removed_children} removed children.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> ClarificationTreeSample:
        sample = self.samples[index]
        
        tree_index = sample["tree_idx"]
        parent_node_idx = sample["parent_node_idx"]
        child_node_idxs = sample["child_node_idxs"]

        if tree_index not in self.cached_reward_tree_idxs:
            assert self.cfg is not None, "Config must be provided to compute rewards"
            self.sidecars[tree_index].compute_rewards(self.cfg)
            self.cached_reward_tree_idxs.add(tree_index)

        tree = self.trees[tree_index]
        sidecar = self.sidecars[tree_index]

        child_advantages = [sidecar.advantage_cache[child_node_idx] for child_node_idx in child_node_idxs]

        # extract the logits for the child nodes
        # child_logits = [sidecar.token_logprobs[child_node_idx] for child_node_idx in child_node_idxs]
        child_token_values = [torch.Tensor([token_tuple[0] for token_tuple in sidecar.token_logprobs[child_node_idx]]) for child_node_idx in child_node_idxs]
        child_token_logprobs = [torch.Tensor([token_tuple[1] for token_tuple in sidecar.token_logprobs[child_node_idx]]) for child_node_idx in child_node_idxs]

        return ClarificationTreeSample(
            tree=tree,
            tree_sidecar=sidecar,
            parent_node_idx=parent_node_idx,
            child_node_idxs=child_node_idxs,
            advantages=child_advantages,
            token_values=child_token_values,
            token_logprobs=child_token_logprobs
        )

@dataclass
class SFTClarificationTreeSample:
    trajectory: DialogTrajectory
    target: str
    advantage: float

class SFTClarificationTreeDataset(Dataset):
    trees: list[DialogTree]
    sidecars: list[TreeSidecar]
    samples: list[dict]
    
    def __init__(self,
        cfg: schema.Config | None,
        trees_path: Path | None = GENERATED_TREES_PATH,
        tree_paths: list[Path] | None = None,
        load_images: bool = True,
        advantage_threshold: float | None = None,
        top_n: int | None = None,
    ):
        self.cfg = cfg
        self.trees_path = trees_path
        self.tree_paths = tree_paths
        self.load_images = load_images
        self.advantage_threshold = advantage_threshold
        self.top_n = top_n

        assert self.trees_path is None or self.tree_paths is None, "One of trees_path or tree_paths must be None"
        assert self.trees_path is not None or self.tree_paths is not None, "One of trees_path or tree_paths must not be None"

        self._load_trees()

    def _load_trees(self):
        trees = []
        sidecars = []
        samples = []
        total_possible_samples = 0

        if self.trees_path is not None:
            tree_dirs = list(self.trees_path.iterdir())
        elif self.tree_paths is not None:
            tree_dirs = self.tree_paths
        else:
            raise ValueError(f"Both trees_path and tree_path")

        for tree_dir in tqdm(tree_dirs, desc="Loading trees for SFT"):
            if not tree_dir.is_dir():
                print(f"Skipping non-directory {tree_dir}")
                continue

            tree_path = tree_dir / "tree.json"
            if not tree_path.exists():
                print(f"Skipping non-tree {tree_path}")
                continue
            
            sidecar_path = tree_dir / "tree_sidecar.json"
            if not sidecar_path.exists():
                print(f"Skipping non-sidecar {sidecar_path}")
                continue
            
            tree = DialogTree.load(tree_path, load_images=self.load_images)
            sidecar = TreeSidecar.load(sidecar_path)
            
            tree_idx = len(trees)
            trees.append(tree)
            sidecars.append(sidecar)
            
            assert self.cfg is not None, "Config must be provided to compute rewards for filtering"
            sidecar.compute_rewards(self.cfg)

            for parent_node_idx, parent_node in tree.get_nodes():
                child_cq_idxs = tree.get_children_idxs(parent_node_idx, type_filter=NodeType.CLARIFICATION_QUESTION)

                if len(child_cq_idxs) == 0:
                    continue

                child_advantages = [(child_idx, sidecar.advantage_cache[child_idx]) for child_idx in child_cq_idxs]
                total_possible_samples += len(child_advantages)
                
                if self.advantage_threshold is not None:
                    child_advantages = [x for x in child_advantages if x[1] >= self.advantage_threshold]
                    
                for child_idx, advantage in child_advantages:
                    samples.append({
                        "tree_idx": tree_idx,
                        "parent_node_idx": parent_node_idx,
                        "target_node_idx": child_idx,
                        "advantage": advantage
                    })

        if self.top_n is not None:
            samples = sorted(samples, key=lambda x: x["advantage"], reverse=True)[:self.top_n]

        self.trees = trees
        self.sidecars = sidecars
        self.samples = samples

        pct_used = (len(self.samples) / total_possible_samples * 100) if total_possible_samples > 0 else 0.0
        print(f"Found {len(self.trees)} trees. Total unfiltered samples: {total_possible_samples}. Using {len(self.samples)} SFT samples ({pct_used:.1f}%).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> SFTClarificationTreeSample:
        sample = self.samples[index]
        tree = self.trees[sample["tree_idx"]]
        
        trajectory = tree.get_trajectory(sample["parent_node_idx"])
        target = tree.get_node(sample["target_node_idx"]).response
        
        return SFTClarificationTreeSample(
            trajectory=trajectory,
            target=target,
            advantage=sample["advantage"]
        )


if __name__ == "__main__":
    # ds = ClearVQADataset(load_images=True)
    # print(ds[0])

    import json
    ds = ClarificationTreeDataset(cfg=None, precompute_rewards=False)
    print(f"Dataset size: {len(ds)}")
    sample = ds[1000]

    traj = sample.tree.get_trajectory(sample.parent_node_idx)
    messages = traj.to_messages("qwen-3-vl", use_img_path=True)
    print(json.dumps(messages, indent=2))

    for i, child_node_idx in enumerate(sample.child_node_idxs):
        child_node = sample.tree.get_node(child_node_idx)
        print(f"  (Adv: {sample.advantages[i]}) {child_node}")