#llama-server -m MODEL.gguf --ctx-size 65536 --port 8002 --chat-template-kwargs '{"enable_thinking": false}' --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0.0 --presence-penalty 1.5 --repeat-penalty 1.0


import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), 'keys.env'))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
models = {
    "XS": {
        "model": "Qwen3.5-4B-Q6_K",
        "provider": "openai",
        "base_url": "http://localhost:8001/v1",
        "api_key": "local"
    },
    "XS_UNSAFE": {
        "model": "Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q6_K",
        "provider": "openai",
        "base_url": "http://localhost:8001/v1",
        "api_key": "local"
    },
    "S": {
        "model": "Qwen3.5-9B-Q6_K",
        "provider": "openai",
        "base_url": "http://localhost:8001/v1",
        "api_key": "local"
    },
    "S_UNSAFE": {
        "model": "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q6_K",
        "provider": "openai",
        "base_url": "http://localhost:8001/v1",
        "api_key": "local"
    },
    "M": {
        "model": "Qwen3.5-35B-A3B-Q6_K",
        "provider": "openai",
        "base_url": "http://localhost:8001/v1",
        "api_key": "local"
    },
    "M_UNSAFE": {
        "model": "Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q6_K",
        "provider": "openai",
        "base_url": "http://localhost:8001/v1",
        "api_key": "local"
    },
    "L": {
        "model": "Qwen3.5-122B-A10B-Q4_K_S",
        "provider": "openai",
        "base_url": "http://localhost:8001/v1",
        "api_key": "local"
    },
    "XL": {
        "model": "gpt-5-2025-08-07",
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "api_key": OPENAI_API_KEY
        }
}

