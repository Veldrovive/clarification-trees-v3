import asyncio
import traceback
import random
import uuid
import numpy as np
from pathlib import Path
from codetiming import Timer
from rich.tree import Tree
from rich import print
from tqdm import tqdm

from clarification_trees_v3.models import BidirectionalEntailmentClusterer
from clarification_trees_v3.utils import add_inference_messages, add_cq_messages, add_answer_messages, SentenceAnalyzer
from clarification_trees_v3.dataset.dataset import ClearVQADataset
from clarification_trees_v3.dataset.dialog_tree import DialogTree, NodeType, DialogTrajectory, TreeSidecar
from clarification_trees_v3.models.remote_vllm_model import RemoteVLLMModel
from clarification_trees_v3.config.schema import Config

class DialogTreeDFSManager:
    def __init__(self, dialog_tree: DialogTree, max_depth: int):
        self.dialog_tree = dialog_tree
        self.max_depth = max_depth
        
        # Stack for DFS containing node indices. 
        # Initialize with the Root node (always index 0 in DialogTree init).
        self.stack = [DialogTree.ROOT]
        
        # Track depth of every node index to manage stopping criteria.
        # Root is at depth 0.
        self.node_depths = {DialogTree.ROOT: 0}

    def has_open_nodes(self) -> bool:
        return len(self.stack) > 0

    def get_next_node(self) -> tuple[DialogTrajectory, NodeType, int]:
        # DFS: Pop from the end (LIFO)
        node_id = self.stack.pop()
        
        # Retrieve node data from the tree.
        # DialogTree.nodes is a list of (parent_idx, DialogNode)
        parent_idx, node = self.dialog_tree.nodes[node_id]
        
        # Construct the trajectory context for this node
        trajectory = self.dialog_tree.get_trajectory(node_id)
        
        return trajectory, node.node_type, node_id

    def add_children(self, parent_node_id: int, new_node_texts: list[str], new_node_trans_probs: list[float], output_node_type: NodeType) -> list[int]:
        parent_depth = self.node_depths[parent_node_id]
        
        # Determine the depth of the new nodes.
        # Logic: 
        # ROOT (0) -> CQ (0) -> CA (1) -> CQ (1) -> CA (2)
        # Depth increases only when an Answer completes a pair.
        new_depth = parent_depth
        if output_node_type == NodeType.CLARIFYING_ANSWER:
            new_depth += 1
            
        # Iterate in reverse so that the first child (index 0) is pushed last 
        # and therefore popped first, maintaining order for the "first" cluster.
        new_node_ids = []
        for text, trans_prob in reversed(list(zip(new_node_texts, new_node_trans_probs))):
            # Add node to the data structure
            # Note: Generated nodes (CQ/CA) do not have images attached, so image=None.
            new_node_idx = self.dialog_tree.add_node(
                parent_idx=parent_node_id,
                node_type=output_node_type,
                image=None,
                response=text,
                transition_prob=trans_prob
            )
            
            # Record depth
            self.node_depths[new_node_idx] = new_depth
            
            # Decide if we should push this node to the stack for further expansion.
            # We stop expanding if we just completed a pair (Answer) and hit max_depth.
            # If we just added a Question, we always expand (to get the Answer).
            should_expand = True
            if output_node_type == NodeType.CLARIFYING_ANSWER and new_depth >= self.max_depth:
                should_expand = False
            
            if should_expand:
                self.stack.append(new_node_idx)
            
            new_node_ids.append(new_node_idx)
        
        return new_node_ids[::-1]


class DialogTreeMCTSManager:
    def __init__(self, dialog_tree: DialogTree, min_iterations: int, max_iterations: int, exploration_constant: float, advantage_threshold: float):
        self.dialog_tree = dialog_tree
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
        self.exploration_constant = exploration_constant
        self.advantage_threshold = advantage_threshold
        
        self.visits = {DialogTree.ROOT: 0}
        self.dead_nodes = set()
        self.iterations_done = 0

    def is_done(self) -> bool:
        if self.iterations_done >= self.max_iterations:
            return True
        if DialogTree.ROOT in self.dead_nodes:
            return True
        return False

    def get_next_leaf_to_expand(self, sidecar: TreeSidecar) -> tuple[DialogTrajectory, NodeType, int, list[int]]:
        current_node_id = DialogTree.ROOT
        path = [current_node_id]
        
        while True:
            cq_children = self.dialog_tree.get_children_idxs(current_node_id, type_filter=NodeType.CLARIFICATION_QUESTION)
            if not cq_children:
                break
            
            candidate_cas = []
            for cq_id in cq_children:
                candidate_cas.extend(self.dialog_tree.get_children_idxs(cq_id, type_filter=NodeType.CLARIFYING_ANSWER))
            
            valid_candidates = [ca for ca in candidate_cas if ca not in self.dead_nodes]
            if not valid_candidates:
                self.dead_nodes.add(current_node_id)
                if current_node_id == DialogTree.ROOT:
                    return self.dialog_tree.get_trajectory(current_node_id), self.dialog_tree.get_node(current_node_id).node_type, current_node_id, path
                return self.get_next_leaf_to_expand(sidecar)

            unexpanded_candidates = [ca for ca in valid_candidates if self.visits.get(ca, 0) == 0]
            
            best_ca = None
            best_ucb = -np.inf
            
            if unexpanded_candidates:
                best_ca = min(unexpanded_candidates, key=lambda ca: sidecar.inference_scores.get(ca, 0))
            else:
                n_parent = self.visits.get(current_node_id, 0)
                for ca in valid_candidates:
                    n_child = self.visits.get(ca, 0)
                    ca_cq_children = self.dialog_tree.get_children_idxs(ca, type_filter=NodeType.CLARIFICATION_QUESTION)
                    if not ca_cq_children:
                        adv_range = 0
                    else:
                        rewards = [sidecar.reward_cache.get(cq, 0) for cq in ca_cq_children if cq in sidecar.reward_cache]
                        adv_range = (max(rewards) - min(rewards)) if rewards else 0
                    
                    ucb = adv_range + self.exploration_constant * np.sqrt(np.log(n_parent) / n_child) if n_child > 0 else np.inf
                    if ucb > best_ucb:
                        best_ucb = ucb
                        best_ca = ca
            
            if best_ca not in unexpanded_candidates and self.iterations_done >= self.min_iterations:
                ca_cq_children = self.dialog_tree.get_children_idxs(best_ca, type_filter=NodeType.CLARIFICATION_QUESTION)
                if not ca_cq_children:
                    adv_range = 0
                else:
                    rewards = [sidecar.reward_cache.get(cq, 0) for cq in ca_cq_children if cq in sidecar.reward_cache]
                    adv_range = (max(rewards) - min(rewards)) if rewards else 0
                    
                if adv_range < self.advantage_threshold:
                    self.dead_nodes.add(best_ca)
                    continue
            
            current_node_id = best_ca
            path.append(current_node_id)
            
        trajectory = self.dialog_tree.get_trajectory(current_node_id)
        node = self.dialog_tree.get_node(current_node_id)
        return trajectory, node.node_type, current_node_id, path

    def update_visits(self, path: list[int]):
        for node_id in path:
            self.visits[node_id] = self.visits.get(node_id, 0) + 1
        self.iterations_done += 1

async def expand_tree(
    cfg: Config,
    tree: DialogTree,
    clusterer: BidirectionalEntailmentClusterer,
    cq_model: RemoteVLLMModel,
    answer_model: RemoteVLLMModel,
    sentence_analyzer: SentenceAnalyzer,
    out_dir: Path,
    seed: int | None = None
) -> tuple[Path, Path]:
    try:
        local_random = random.Random(seed) if seed is not None else random
        dialog_tree_config = cfg.dialog_tree
        max_depth = dialog_tree_config.max_depth
        question_expansion_factor = dialog_tree_config.question_expansion_factor
        answer_expansion_factor = dialog_tree_config.answer_expansion_factor
        question_diverse_sample_count = dialog_tree_config.question_diverse_sample_count
        answer_diverse_sample_count = dialog_tree_config.answer_diverse_sample_count
        inference_diverse_sample_count = dialog_tree_config.inference_diverse_sample_count

        dialog_tree_manager = DialogTreeDFSManager(tree, max_depth=max_depth)

        cq_node_types = set([NodeType.ROOT, NodeType.CLARIFYING_ANSWER])  # Node types that cause the tree to be expanded using the cq model
        answer_node_types = set([NodeType.CLARIFICATION_QUESTION])  # Node types that cause the tree to be expanded using the answer model

        async def _generate_inference(tree: DialogTree, answer_node_ids: list[int], n_outputs: int):
            with Timer("tree/generate_inference", logger=None):
                for answer_node_id in answer_node_ids:
                    dialog_trajectory = tree.get_trajectory(answer_node_id)
                    messages = dialog_trajectory.to_messages(model_name="qwen-3-vl", use_img_path=True)
                    add_inference_messages(messages, model_cfg=cfg.answer_model)

                    with Timer("tree/generate_inference/generate", logger=None):
                        request_output = await answer_model.generate(messages, n_outputs=n_outputs, use_lora=False, seed=seed)
                    generated_texts = [o.message.content for o in request_output.choices if o.message.content is not None]

                    with Timer("tree/generate_inference/cluster", logger=None):
                        clusters, exemplars, _, _ = await clusterer.async_cluster(generated_texts)

                    probabilities = [len(cluster) / len(generated_texts) for cluster in clusters]

                    for exemplar, probability in zip(exemplars, probabilities):
                        tree.add_node(
                            parent_idx=answer_node_id,
                            node_type=NodeType.INFERENCE,
                            response=exemplar,
                            transition_prob=probability
                        )

        tree_save_path = out_dir / f"tree.json"
        sidecar_save_path = out_dir / f"tree_sidecar.json"
        sidecar = TreeSidecar(tree_save_path)
        with Timer("tree", logger=None):
            # We always start by making an inference from the root node.
            await _generate_inference(tree, [DialogTree.ROOT], inference_diverse_sample_count)

        async def _generate_children(node_id: int, input_node_type: NodeType) -> list[int]:
            dialog_trajectory = tree.get_trajectory(node_id)
            messages = dialog_trajectory.to_messages(model_name="qwen-3-vl", use_img_path=True)

            if input_node_type in cq_node_types:
                add_cq_messages(messages, model_cfg=cfg.clarification_model)
                engine = cq_model
                sample_count = question_diverse_sample_count
                expansion_factor = question_expansion_factor
                output_node_type = NodeType.CLARIFICATION_QUESTION
                use_lora = cfg.clarification_model.lora_config is not None and cfg.clarification_model.lora_config.use_lora
                timer_key = "generate_cq"
            elif input_node_type in answer_node_types:
                assert tree.unambiguous_question is not None
                assert tree.answers is not None
                add_answer_messages(messages, unambiguous_question=tree.unambiguous_question, answers=tree.answers, model_cfg=cfg.answer_model)
                engine = answer_model
                sample_count = answer_diverse_sample_count
                expansion_factor = answer_expansion_factor
                output_node_type = NodeType.CLARIFYING_ANSWER
                use_lora = cfg.answer_model.lora_config is not None and cfg.answer_model.lora_config.use_lora
                timer_key = "generate_ca"
            else:
                raise ValueError(f"Unknown node type: {input_node_type}")

            with Timer(f"tree/{timer_key}", logger=None):
                with Timer(f"tree/{timer_key}/generate", logger=None):
                    request_output = await engine.generate(messages, n_outputs=sample_count, use_lora=use_lora, use_tokens_as_ids=True, logprobs=True, seed=seed)
                    valid_choices = [choice for choice in request_output.choices if choice.message.content is not None]
                    generated_texts = [choice.message.content for choice in valid_choices]
                    generated_logprobs = [
                        [(int(o.token.split(":")[1]), o.logprob) for o in choice.logprobs.content]
                        if choice.logprobs and choice.logprobs.content else []
                        for choice in valid_choices
                    ]

                with Timer(f"tree/{timer_key}/cluster", logger=None):
                    clusters, exemplars, metadata_clusters, metadata_exemplars = await clusterer.async_cluster(generated_texts, generated_logprobs)

            if not clusters:
                return []
            if len(clusters) > expansion_factor:
                cluster_indices = local_random.sample(range(len(clusters)), expansion_factor)
                clusters = [clusters[i] for i in cluster_indices]
                exemplars = [exemplars[i] for i in cluster_indices]
                metadata_clusters = [metadata_clusters[i] for i in cluster_indices]
                metadata_exemplars = [metadata_exemplars[i] for i in cluster_indices]
            total_allowed_texts = sum([len(cluster) for cluster in clusters])
            probabilities = [len(cluster) / total_allowed_texts for cluster in clusters]

            new_node_ids = []
            for text, trans_prob in zip(exemplars, probabilities):
                new_node_idx = tree.add_node(
                    parent_idx=node_id,
                    node_type=output_node_type,
                    image=None,
                    response=text,
                    transition_prob=trans_prob
                )
                new_node_ids.append(new_node_idx)

            for new_node_id, metadata_exemplar in zip(new_node_ids, metadata_exemplars):
                sidecar.add_logprobs(new_node_id, metadata_exemplar)

            return new_node_ids

        if getattr(cfg.dialog_tree, "algorithm", "dfs") == "dfs":
            while dialog_tree_manager.has_open_nodes():
                dialog_trajectory, input_node_type, node_id = dialog_tree_manager.get_next_node()
                new_node_ids = await _generate_children(node_id, input_node_type)
                if not new_node_ids:
                    continue
                # Push children to stack
                parent_depth = dialog_tree_manager.node_depths[node_id]
                output_node_type = tree.get_node(new_node_ids[0]).node_type
                new_depth = parent_depth + (1 if output_node_type == NodeType.CLARIFYING_ANSWER else 0)
                should_expand = True
                if output_node_type == NodeType.CLARIFYING_ANSWER and new_depth >= dialog_tree_manager.max_depth:
                    should_expand = False
                for new_node_id in reversed(new_node_ids):
                    dialog_tree_manager.node_depths[new_node_id] = new_depth
                    if should_expand:
                        dialog_tree_manager.stack.append(new_node_id)
                        
                if output_node_type == NodeType.CLARIFYING_ANSWER:
                    await _generate_inference(tree, new_node_ids, inference_diverse_sample_count)
        elif cfg.dialog_tree.algorithm == "mcts":
            mcts_manager = DialogTreeMCTSManager(
                tree,
                min_iterations=cfg.dialog_tree.mcts_min_iterations,
                max_iterations=cfg.dialog_tree.mcts_max_iterations,
                exploration_constant=cfg.dialog_tree.mcts_exploration_constant,
                advantage_threshold=cfg.dialog_tree.mcts_advantage_threshold
            )
            while not mcts_manager.is_done():
                trajectory, node_type, leaf_node_id, path = mcts_manager.get_next_leaf_to_expand(sidecar)
                if leaf_node_id in mcts_manager.dead_nodes:
                    break # Root is dead
                
                new_cq_ids = await _generate_children(leaf_node_id, NodeType.ROOT if node_type == NodeType.ROOT else NodeType.CLARIFYING_ANSWER)
                all_new_ca_ids = []
                for cq_id in new_cq_ids:
                    new_ca_ids = await _generate_children(cq_id, NodeType.CLARIFICATION_QUESTION)
                    all_new_ca_ids.extend(new_ca_ids)
                
                if all_new_ca_ids:
                    await _generate_inference(tree, all_new_ca_ids, inference_diverse_sample_count)
                
                await sidecar.compute_all_scores(answer_model, sentence_analyzer, clusterer, cfg, tree)
                sidecar.compute_rewards(tree)
                mcts_manager.update_visits(path)

        tree.save(tree_save_path)
        with Timer("reward", logger=None):
            await sidecar.compute_all_scores(answer_model, sentence_analyzer, clusterer, cfg, tree)
            sidecar.save(sidecar_save_path)

        return tree_save_path, sidecar_save_path
    except Exception as e:
        traceback.print_exc()
        raise e


async def process_dataset_lazily(
    cfg: Config,
    dataset: ClearVQADataset,
    indices: list[int] | None,
    clusterer: BidirectionalEntailmentClusterer,
    cq_model: RemoteVLLMModel,
    answer_model: RemoteVLLMModel,
    sentence_analyzer: SentenceAnalyzer,
    out_dir: Path,
    N_parallel_trees: int = 10,
    tqdm_position: int = 0,
    tqdm_desc: str = "Expanding Trees"
):
    # Configuration
    if indices is None:
        indices = list(range(len(dataset)))
    
    total_items = len(indices)
    
    # We keep a set of currently running tasks
    active_tasks = set()
    
    # Initialize progress bar
    pbar = tqdm(total=total_items, desc=tqdm_desc, position=tqdm_position, smoothing=0.0)

    n_done = 0
    iterator = iter(indices)

    try:
        while True:
            # 1. THROTTLING: Load more tasks up to N_parallel_trees
            while len(active_tasks) < N_parallel_trees:
                try:
                    i = next(iterator)
                except StopIteration:
                    break
                
                # LAZY LOADING: The image is loaded into memory HERE, not before.
                sample = dataset[i]
                tree = DialogTree(
                    init_question=sample.blurred_question,
                    init_image=None,
                    init_image_path=sample.image_path,
                    init_image_caption=sample.caption,
                    unambiguous_question=sample.question,
                    gold_answer=sample.gold_answer,
                    answers=sample.answers
                )

                # DISPATCH: Create the coroutine and track it
                img_path = tree.init_image_path
                if img_path is None:
                    out_dir_i = out_dir / f"tree_{uuid.uuid4()}"
                else:
                    img_name = img_path.stem
                    out_dir_i = out_dir / f"tree_{img_name}_{uuid.uuid4()}"
                out_dir_i.mkdir(parents=True, exist_ok=True)
                task = asyncio.create_task(
                    expand_tree(
                        cfg=cfg,
                        tree=tree,
                        clusterer=clusterer,
                        cq_model=cq_model,
                        answer_model=answer_model,
                        sentence_analyzer=sentence_analyzer,
                        out_dir=out_dir_i,
                        seed=getattr(cfg, 'seed', 42) + i
                    )
                )
                active_tasks.add(task)

            if not active_tasks:
                break

            # 2. Wait for at least one task to finish
            done, pending = await asyncio.wait(
                active_tasks, 
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # 3. CLEANUP: Process finished tasks and remove from active set
            for task in done:
                try:
                    finished_tree_paths, finished_sidecar_path = await task # Retrieve the result
                    yield finished_tree_paths, finished_sidecar_path
                except Exception as e:
                    print(f"Task failed: {e}")
                finally:
                    pbar.update(1)
                    n_done += 1

                    if n_done % N_parallel_trees == 0:
                        print(f"\n\nTimer breakdown after {n_done} trees:")
                        print_timer_tree()
            
            # Update active_tasks to only contain the ones still running
            active_tasks = pending

    finally:
        # 4. DRAIN/CANCEL: Wait for remaining tasks to be cancelled if exited early
        for task in active_tasks:
            task.cancel()
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)
        pbar.close()

def print_timer_tree():
    data = Timer.timers

    if "total" in data:
        total_time = data.pop("total")
        root = Tree(f"[bold cyan]Total[/]: [yellow]{total_time:.2f}[/]")
    else:
        total_time = None
        root = Tree("Root", hide_root=True)
    
    # We keep a map of path -> tree_node to avoid rebuilding branches
    # We strip the dummy root logic for simplicity by sorting keys
    path_map = {}

    for key, value in sorted(data.items()):
        parts = key.split('/')
        
        # Determine the leaf name and formatting
        name = parts[-1]
        if total_time is not None:
            label = f"[bold cyan]{name}[/]: [yellow]{value:.2f}[/] [dim]({value/total_time*100:.1f}%)[/dim]"
        else:
            label = f"[bold cyan]{name}[/]: [yellow]{value:.2f}[/]"
        
        parent_path = "/".join(parts[:-1])
        
        if parent_path in path_map:
            # Add to existing parent
            node = path_map[parent_path].add(label)
        else:
            # This is a top-level node (like 'tree' or 'reward')
            node = root.add(label)
            
        path_map[key] = node

    print(root)
