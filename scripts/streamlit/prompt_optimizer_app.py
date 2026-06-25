import streamlit as st
import asyncio
import os
import glob
from pathlib import Path
from omegaconf import OmegaConf
from hydra import compose, initialize

import sys
# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))

from clarification_trees_v3.config.schema import parse_config
from clarification_trees_v3.models.utils import use_models
from clarification_trees_v3.dataset.dataset import ClearVQADataset
from clarification_trees_v3.dataset.dialog_tree import DialogTree, NodeType
from clarification_trees_v3.utils import add_cq_messages, add_answer_messages, add_inference_messages

st.set_page_config(layout="wide", page_title="Prompt Optimizer & Debugger")

# --- Async Utility ---
@st.cache_resource
def get_async_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop

def run_async(coro):
    loop = get_async_loop()
    return loop.run_until_complete(coro)

from hydra.core.global_hydra import GlobalHydra

def apply_overrides(cfg):
    config_dir = Path(__file__).resolve().parent.parent.parent / "config"
    overrides_file = config_dir / "prompt_overrides.yaml"
    if not overrides_file.exists():
        return cfg
        
    all_overrides = OmegaConf.load(overrides_file)
    
    # We need to know which files are used in evaluate_mcts_c.yaml
    eval_cfg_raw = OmegaConf.load(config_dir / "evaluate_mcts_c.yaml")
    defaults = eval_cfg_raw.get("defaults", [])
    
    file_to_config_key = {}
    for d in defaults:
        if isinstance(d, str):
            continue
        for k, v in d.items():
            if k == "clarification_model/base":
                file_to_config_key[f"{v}.yaml"] = "clarification_model"
            elif k == "answer_model/base":
                file_to_config_key[f"{v}.yaml"] = "answer_model"
                
    for file_name, config_key in file_to_config_key.items():
        if file_name in all_overrides:
            file_overrides = all_overrides[file_name]
            target_cfg = getattr(cfg, config_key)
            for k, v in file_overrides.items():
                if k == "judge_prompts" and hasattr(target_cfg, "judge_prompts"):
                    if "base_prompt" in v:
                        target_cfg.judge_prompts.base_prompt = v["base_prompt"]
                    if "instruction_prompt" in v:
                        target_cfg.judge_prompts.instruction_prompt = v["instruction_prompt"]
                elif hasattr(target_cfg, k):
                    setattr(target_cfg, k, v)
    return cfg

# --- Resource Loading ---
@st.cache_resource
def load_models():
    loop = get_async_loop()
    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="../../config", job_name="streamlit_app_init"):
        raw_cfg = compose(config_name="evaluate_mcts_c")
        cfg = parse_config(raw_cfg)
    
    cfg = apply_overrides(cfg)
        
    ctx = use_models(cfg)
    cq_model, answer_model = loop.run_until_complete(ctx.__aenter__())
    return cq_model, answer_model

def get_config():
    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="../../config", job_name="streamlit_app_cfg"):
        raw_cfg = compose(config_name="evaluate_mcts_c")
        cfg = parse_config(raw_cfg)
        return apply_overrides(cfg)

@st.cache_resource
def load_dataset():
    return ClearVQADataset(load_images=False, table_name="val_annotated.jsonl")

# --- UI Sidebar ---
page = st.sidebar.radio("Navigation", ["Prompt Editor", "Generation Stepper"])

# --- Page 1: Prompt Editor ---
if page == "Prompt Editor":
    st.title("Prompt Editor")
    
    config_dir = Path(__file__).resolve().parent.parent.parent / "config"
    
    # Discover yaml files
    answer_models = list((config_dir / "answer_model" / "base").glob("*.yaml"))
    clarification_models = list((config_dir / "clarification_model" / "base").glob("*.yaml"))
    
    all_files = answer_models + clarification_models
    file_options = {f.name: f for f in all_files}
    
    selected_file_name = st.selectbox("Select Config File", list(file_options.keys()))
    
    if selected_file_name:
        selected_file_path = file_options[selected_file_name]
        st.write(f"Editing: `{selected_file_path}`")
        
        # Load yaml and overlay overrides
        conf = OmegaConf.load(selected_file_path)
        overrides_file = config_dir / "prompt_overrides.yaml"
        if overrides_file.exists():
            all_overrides = OmegaConf.load(overrides_file)
            if selected_file_name in all_overrides:
                file_overrides = all_overrides[selected_file_name]
                for k, v in file_overrides.items():
                    if k == "judge_prompts" and "judge_prompts" in conf:
                        if "base_prompt" in v:
                            conf.judge_prompts.base_prompt = v["base_prompt"]
                        if "instruction_prompt" in v:
                            conf.judge_prompts.instruction_prompt = v["instruction_prompt"]
                    elif k in conf:
                        conf[k] = v
        
        def save_prompt(file_path, config_key):
            overrides_file = config_dir / "prompt_overrides.yaml"
            if overrides_file.exists():
                all_overrides = OmegaConf.load(overrides_file)
            else:
                all_overrides = OmegaConf.create()
                
            file_name = file_path.name
            if file_name not in all_overrides:
                all_overrides[file_name] = {}
                
            state_key = f"editor_{file_name}_{config_key}"
            
            if config_key == "judge_prompts_base":
                if "judge_prompts" not in all_overrides[file_name]:
                    all_overrides[file_name]["judge_prompts"] = {}
                all_overrides[file_name]["judge_prompts"]["base_prompt"] = st.session_state[state_key]
            elif config_key == "judge_prompts_inst":
                if "judge_prompts" not in all_overrides[file_name]:
                    all_overrides[file_name]["judge_prompts"] = {}
                all_overrides[file_name]["judge_prompts"]["instruction_prompt"] = st.session_state[state_key]
            else:
                all_overrides[file_name][config_key] = st.session_state[state_key]
                
            OmegaConf.save(all_overrides, overrides_file)

        prompt_keys = [
            "base_prompt",
            "answer_base_prompt",
            "answer_instruction_prompt",
            "inference_base_prompt",
            "inference_instruction_prompt",
        ]
        
        for k, v in conf.items():
            if k in prompt_keys and isinstance(v, str):
                state_key = f"editor_{selected_file_name}_{k}"
                if state_key not in st.session_state:
                    st.session_state[state_key] = v
                # If the value on disk changed externally, we might want to update session state, but we'll assume Streamlit is the main editor.
                st.text_area(f"{k}", key=state_key, height=200, on_change=save_prompt, args=(selected_file_path, k))
            elif k == "judge_prompts":
                st.subheader("Judge Prompts")
                
                base_key = f"editor_{selected_file_name}_judge_prompts_base"
                if base_key not in st.session_state:
                    st.session_state[base_key] = conf.judge_prompts.base_prompt
                st.text_area("judge_prompts.base_prompt", key=base_key, height=200, on_change=save_prompt, args=(selected_file_path, "judge_prompts_base"))
                
                inst_key = f"editor_{selected_file_name}_judge_prompts_inst"
                if inst_key not in st.session_state:
                    st.session_state[inst_key] = conf.judge_prompts.instruction_prompt
                st.text_area("judge_prompts.instruction_prompt", key=inst_key, height=200, on_change=save_prompt, args=(selected_file_path, "judge_prompts_inst"))

# --- Page 2: Generation Stepper ---
elif page == "Generation Stepper":
    st.title("Generation Stepper")
    
    cq_model, answer_model = load_models()
    cfg = get_config()
    dataset = load_dataset()
    
    # Initialize state
    if "tree" not in st.session_state:
        st.session_state.tree = None
    if "current_node_id" not in st.session_state:
        st.session_state.current_node_id = DialogTree.ROOT
    if "next_step_type" not in st.session_state:
        st.session_state.next_step_type = NodeType.CLARIFICATION_QUESTION
    if "log_entries" not in st.session_state:
        st.session_state.log_entries = []
    if "raw_inputs" not in st.session_state:
        st.session_state.raw_inputs = None

    tree_index = st.number_input("Tree Index", min_value=0, max_value=len(dataset)-1, value=0)
    
    if st.button("Load Tree"):
        sample = dataset[tree_index]
        st.session_state.tree = DialogTree(
            init_question=sample.blurred_question,
            init_image=None,
            init_image_path=sample.image_path,
            init_image_caption=sample.caption,
            unambiguous_question=sample.question,
            gold_answer=sample.gold_answer,
            answers=sample.answers
        )
        st.session_state.current_node_id = DialogTree.ROOT
        st.session_state.next_step_type = NodeType.CLARIFICATION_QUESTION
        st.session_state.log_entries = []
        st.session_state.raw_inputs = None
        st.rerun()
        
    if st.session_state.tree is not None:
        tree = st.session_state.tree
        st.write(f"**Blurred Question:** {tree.init_question}")
        st.write(f"**Unambiguous Question:** {tree.unambiguous_question}")
        st.write(f"**Answers:** {tree.answers}")
        st.write(f"**Gold Answer:** {tree.gold_answer}")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Output Log")
            for entry in st.session_state.log_entries:
                st.markdown(f"**{entry['type'].name}**: {entry['text']}")
                
            st.write("---")
            if st.button(f"Step Generation: {st.session_state.next_step_type.name}"):
                node_id = st.session_state.current_node_id
                next_type = st.session_state.next_step_type
                
                dialog_trajectory = tree.get_trajectory(node_id)
                messages = dialog_trajectory.to_messages(model_name="qwen-3-vl", use_img_path=True)
                
                async def generate_step():
                    use_lora_cq = cfg.clarification_model.lora_config is not None and cfg.clarification_model.lora_config.use_lora
                    use_lora_ca = cfg.answer_model.lora_config is not None and cfg.answer_model.lora_config.use_lora
                    
                    if next_type == NodeType.CLARIFICATION_QUESTION:
                        add_cq_messages(messages, model_cfg=cfg.clarification_model)
                        st.session_state.raw_inputs = messages
                        res = await cq_model.generate(messages, n_outputs=1, use_lora=use_lora_cq)
                        
                    elif next_type == NodeType.CLARIFYING_ANSWER:
                        add_answer_messages(messages, unambiguous_question=tree.unambiguous_question, answers=tree.answers, model_cfg=cfg.answer_model)
                        st.session_state.raw_inputs = messages
                        res = await answer_model.generate(messages, n_outputs=1, use_lora=use_lora_ca)
                        
                    elif next_type == NodeType.INFERENCE:
                        add_inference_messages(messages, model_cfg=cfg.answer_model)
                        st.session_state.raw_inputs = messages
                        res = await answer_model.generate(messages, n_outputs=1, use_lora=use_lora_ca)
                        
                    return res.choices[0].message.content
                
                with st.spinner(f"Generating {next_type.name}..."):
                    generated_text = run_async(generate_step())
                
                # Add to tree
                new_node_idx = tree.add_node(
                    parent_idx=node_id,
                    node_type=next_type,
                    response=generated_text,
                    transition_prob=1.0,
                    image=None
                )
                
                st.session_state.log_entries.append({
                    "type": next_type,
                    "text": generated_text
                })
                
                if next_type != NodeType.INFERENCE:
                    st.session_state.current_node_id = new_node_idx
                
                # Update next step type
                if next_type == NodeType.CLARIFICATION_QUESTION:
                    st.session_state.next_step_type = NodeType.CLARIFYING_ANSWER
                elif next_type == NodeType.CLARIFYING_ANSWER:
                    st.session_state.next_step_type = NodeType.INFERENCE
                elif next_type == NodeType.INFERENCE:
                    st.session_state.next_step_type = NodeType.CLARIFICATION_QUESTION
                    
                st.rerun()

        with col2:
            st.subheader("Raw Inputs")
            if st.session_state.raw_inputs:
                st.json(st.session_state.raw_inputs)
            else:
                st.write("No inputs yet. Press step generation.")
