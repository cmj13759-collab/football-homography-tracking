from dotenv import load_dotenv
import os
from pathlib import Path

# Load .env from project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["ROBOFLOW_API_KEY"] = os.getenv("ROBOFLOW_API_KEY")

os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CUDAExecutionProvider]"