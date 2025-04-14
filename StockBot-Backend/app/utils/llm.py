import os
import logging
from llama_cpp import Llama

logger = logging.getLogger(__name__)

model_path = os.path.abspath("models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")

llm = None
llm_enabled = False

try:
    logger.info("Trying to load local LLM model...")
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=4,
        n_gpu_layers=20  # Can tweak this later for M1/M2
    )
    # Do a warmup call
    llm("Hello")
    llm_enabled = True
    logger.info("✅ Local LLM loaded successfully.")
except Exception as e:
    logger.warning("❌ Local LLM disabled. Reason: %s", str(e))
    llm = None
    llm_enabled = False

def local_llm_answer(prompt):
    if not llm_enabled or llm is None:
        raise RuntimeError("Local LLM not available.")
    output = llm(prompt, max_tokens=512, stop=["\n"])
    return output['choices'][0]['text'].strip()
