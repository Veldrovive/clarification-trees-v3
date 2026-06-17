"""
Manages a subprocess that runs a vLLM server
"""

from omegaconf import DictConfig
from pathlib import Path
import requests
import os
import subprocess
import asyncio
import time
import httpx
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
import clarification_trees_v3.config.schema as schema

class RemoteVLLMModel:
    process: subprocess.Popen | None
    client: AsyncOpenAI | None
    is_running_internally: bool
    is_running: bool

    allowed_model_keys: list[str]

    def __init__(
        self,
        model_cfg: schema.ClarificationModelType | schema.AnswerModelType,
        paths_config: schema.PathsConfig,
        loras_path: Path,
        gpus: list[int] = [0],
        max_model_len: int = 4096 * 2,
        gpu_memory_utilization: float = 0.9,
        port: int = 29002,
        startup_timeout: int = 60*50,
        max_lora_rank: int = 64,
        log_file: Path | None = None,
        environment_path: Path | None = None,
        debug: bool = False
    ):
        self.client = None
        self.process = None
        self.is_running_internally = False
        self.is_running = False
        self.allowed_model_keys = []

        self.model_cfg = model_cfg

        self.base_model_path = schema.resolve_base_model_path(model_cfg.base_model_source, paths_config)
        self.lora_config = model_cfg.lora_config
        self.use_lora = self.lora_config.use_lora if self.lora_config else False
        self.lora_id = self.lora_config.lora_id if self.lora_config else None

        if hasattr(model_cfg, "sampling_params") and getattr(model_cfg, "sampling_params") is not None:
            self.sampling_params = model_cfg.sampling_params
        else:
            self.sampling_params = {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 40,
                "repetition_penalty": 1.0,
                "presence_penalty": 2.0,
                "max_tokens": 1024,
                "stop_token_ids": []
            }
        self.loras_path = loras_path
        self.gpus = gpus
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.port = port
        self.startup_timeout = startup_timeout
        self.max_lora_rank = max_lora_rank
        self.log_file = log_file
        self.environment_path = environment_path  # Like "./venv"
        self.debug = debug

        self.process = None
        if self.check_health():
            # Then there is already a server running, so we don't need to start one
            self.is_running_internally = False
            self.is_running = True
            print(f"vLLM server already running on port {self.port}. Skipping startup.")
        else:
            print(f"No vLLM server running on port {self.port}. Server needs to be started manually. Call start_server() to start the server.")

        self.client = self._get_openai_client()

    def _get_base_url(self):
        return f"http://localhost:{self.port}"
    
    def _get_openai_client(self):
        return AsyncOpenAI(
            base_url=self._get_base_url() + "/v1",
            api_key="EMPTY"
        )

    def check_health(self):
        try:
            response = requests.get(f"{self._get_base_url()}/health")
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    async def _get_loaded_model_ids(self):
        assert self.client is not None
        response = await self.client.models.list()
        return [model.id for model in response.data]

    async def unload_lora_adapter(self, lora_id: str):
        current_models = await self._get_loaded_model_ids()
        if lora_id not in current_models:
            raise ValueError(f"LoRA adapter {lora_id} not loaded")

        remove_lora_url = f"{self._get_base_url()}/v1/unload_lora_adapter"
        payload = {
            "lora_name": lora_id
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(remove_lora_url, json=payload, timeout=60)
            if response.status_code != 200:
                raise Exception(f"Failed to unload LoRA adapter {lora_id}: {response.text}")

    async def load_lora_adapter(self, lora_id: str, allow_overwrite: bool = False):
        current_models = await self._get_loaded_model_ids()
        if lora_id in current_models:
            if allow_overwrite:
                # Then we first need to remove the model
                await self.unload_lora_adapter(lora_id)
            else:
                raise ValueError(f"LoRA adapter {lora_id} already loaded")
        
        adapter_path = self.loras_path / lora_id / "best_adapter"
        assert adapter_path.exists(), f"LoRA adapter {lora_id} not found at {adapter_path}"

        add_lora_url = f"{self._get_base_url()}/v1/load_lora_adapter"
        payload = {
            "lora_name": lora_id,
            "lora_path": str(adapter_path)
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(add_lora_url, json=payload, timeout=60)
            if response.status_code != 200:
                raise Exception(f"Failed to load LoRA adapter {lora_id}: {response.text}")
            
            self.allowed_model_keys = await self._get_loaded_model_ids()
    
    async def initialize_server(self):
        if self.is_running:
            self.allowed_model_keys = await self._get_loaded_model_ids()
            print(f"Found existing vLLM server on port {self.port} with models {self.allowed_model_keys}")
            assert self.base_model_path in self.allowed_model_keys, f"Base model {self.base_model_path} not found in vLLM server on port {self.port}"
            if self.use_lora:
                assert self.lora_id in self.allowed_model_keys, f"LoRA adapter {self.lora_id} not found in vLLM server on port {self.port}"
            return

        python_executable = "python"
        if self.environment_path is not None:
            python_executable = str(self.environment_path / "bin" / "python")

        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpus))
        env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "true"
        if self.debug:
            env["VLLM_LOGGING_LEVEL"] = "DEBUG"
            env["CUDA_LAUNCH_BLOCKING"] = "1"
            env["NCCL_DEBUG"] = "TRACE"

        command = [
            # python_executable, "-m", "vllm.entrypoints.openai_api_server", self.base_model_path,
            "uv", "run", "vllm", "serve", self.base_model_path,
            "--host", "0.0.0.0",
            "--port", str(self.port),
            "--trust-remote-code",
            "--tensor-parallel-size", str(len(self.gpus)),
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--allowed-local-media-path", "/",
            "--enable-lora",
            "--max-lora-rank", str(self.max_lora_rank)
        ]

        if self.log_file is not None:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, "w") as f:
                self.process = subprocess.Popen(command, env=env, stdout=f, stderr=f)
        else:
            self.process = subprocess.Popen(command, env=env)
        
        print(f"Started vLLM server with command: {command}")

        print(f"Waiting for vLLM server to start on port {self.port}...")
        start_time = time.time()
        while not self.check_health():
            # Check if the process died
            if self.process.poll() is not None:
                raise Exception(f"vLLM server died with exit code {self.process.returncode}")
            if time.time() - start_time > self.startup_timeout:
                raise Exception(f"vLLM server failed to start on port {self.port}")
            await asyncio.sleep(1)
    
        self.is_running = True
        self.is_running_internally = True
        print(f"vLLM server started on port {self.port}")

        if self.use_lora:
            print(f"Loading LoRA adapter {self.lora_id}")
            await self.load_lora_adapter(self.lora_id)

        self.allowed_model_keys = await self._get_loaded_model_ids()
        assert self.base_model_path in self.allowed_model_keys, f"Base model {self.base_model_path} not found in vLLM server on port {self.port}"
        if self.use_lora:
            assert self.lora_id in self.allowed_model_keys, f"LoRA adapter {self.lora_id} not found in vLLM server on port {self.port}"

    def stop_server(self):
        if self.is_running_internally:
            assert self.process is not None, "Process is None, but is_running_internally is True"
            self.process.terminate()
            self.process.wait(timeout=10)
            self.is_running = False
            self.is_running_internally = False
            print(f"vLLM server stopped on port {self.port}")
        else:
            print(f"Stop called, but vLLM server on port {self.port} is external or not running")

    async def generate(self, messages, n_outputs: int = 1, use_tokens_as_ids: bool = False, logprobs: int | bool | None = None, model_key: str | None = None, use_lora: bool | None = None) -> ChatCompletion:
        sampling_params = {
            **self.sampling_params,
            "return_token_ids": True,
            "n": n_outputs
        }

        if use_tokens_as_ids:
            sampling_params["return_tokens_as_token_ids"] = True

        if logprobs is not None:
            if isinstance(logprobs, bool):
                sampling_params["logprobs"] = logprobs
            elif isinstance(logprobs, int):
                sampling_params["logprobs"] = True
                sampling_params["top_logprobs"] = logprobs
            else:
                raise ValueError(f"logprobs must be a boolean or an integer, got {logprobs}")

        assert model_key is None or use_lora is None, "Cannot specify both model_key and use_lora"
        if model_key is not None:
            chosen_key = model_key
        elif use_lora:
            chosen_key = self.lora_id
        else:
            chosen_key = self.base_model_path

        if chosen_key not in self.allowed_model_keys:
            # Might be that it was added externally. Refresh the list and check again
            self.allowed_model_keys = await self._get_loaded_model_ids()
            
        if chosen_key not in self.allowed_model_keys:
            # We're now confident that the model is not loaded
            raise ValueError(f"Model {chosen_key} not found in vLLM server on port {self.port}")

        assert self.client is not None, "Client is None, but vLLM server is running"
        response = await self.client.chat.completions.create(
            model=chosen_key,
            messages=messages,
            extra_body=sampling_params,
        )

        return response

async def run_test():
    from hydra import compose, initialize
    from omegaconf import OmegaConf
    from clarification_trees.dataset import ClearVQADataset
    from clarification_trees.dialog_tree import DialogTree

    from dotenv import load_dotenv
    load_dotenv()

    ds = ClearVQADataset(load_images=False)
    test_sample = ds[10]
    tree = DialogTree(
        test_sample.blurred_question,
        None,
        test_sample.image_path,
        test_sample.caption,
        test_sample.question,
        test_sample.gold_answer,
        test_sample.answers
    )

    with initialize(version_base=None, config_path="../../config", job_name="test_app"):
        config = compose(config_name="config")
    
    lora_checkpoint_path = Path(config.paths.checkpoints.loras)
    
    remote_model = RemoteVLLMModel(
        config.clarification_model,
        config.paths,
        lora_checkpoint_path,
        environment_path=Path("/scratch4/home/adempst/projects/clarification-trees-v2/venv_vllm"),
        gpus=[6, 7],
        log_file=Path("test_vllm_server.log")
    )
    system_prompt = config.clarification_model.base_prompt

    dialog_traj = tree.get_trajectory(DialogTree.ROOT)
    messages = dialog_traj.to_messages("qwen-3-vl", use_img_path=True)
    messages.insert(0, {"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    print(f"Testing with messages: {messages}")

    await remote_model.initialize_server()
    for model_key in remote_model.allowed_model_keys:
        print(f"Testing with model {model_key}")
        res = await remote_model.generate(messages, 10, model_key=model_key)
        for i, choice in enumerate(res.choices):
            print(f"  Choice {i}: {choice.message.content}")
    
    remote_model.stop_server()

if __name__ == "__main__":
    asyncio.run(run_test())
