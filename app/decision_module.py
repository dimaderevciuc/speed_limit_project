import numpy as np
from pathlib import Path
# Import SpeedLimitNN lazily inside load_nn_model to avoid importing torch at module import time
from app.llm_openai import get_speed_reduction_from_air_quality
import logging

logger = logging.getLogger(__name__)

# Try importing torch but fail gracefully if it's not installed.
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False
    logger.warning("PyTorch is not installed. NeuralNet path will use deterministic fallback.")


def load_nn_model():
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")

    # Resolve model path relative to project root (not CWD)
    base = Path(__file__).resolve().parents[1]
    model_path = base / 'model' / 'speed_limit_nn.pth'

    # Import the network class lazily so missing torch doesn't break module import
    from model.train_nn import SpeedLimitNN

    model = SpeedLimitNN()
    # Load onto CPU by default for portability
    state = torch.load(str(model_path), map_location=torch.device('cpu'))
    model.load_state_dict(state)
    model.eval()
    return model


def calculate_speed_limit(weather, darkness, air_quality_index):
    BASE_SPEED = 80

    # If bad weather or dark: try NN prediction, otherwise fall back safely
    if darkness or weather == "bad":
        if not TORCH_AVAILABLE:
            logger.warning("Requested NN path but PyTorch unavailable — using deterministic fallback speed 30 km/h")
            return 30, 'fallback'

        try:
            model = load_nn_model()
            X = np.array([[80, 60, 25, 110, 10000]], dtype=np.float32)
            with torch.no_grad():
                # Ensure torch tensor dtype and extract scalar with .item()
                inp = torch.tensor(X, dtype=torch.float32)
                out = model(inp)
                reduction = float(out.item())
            return max(30, BASE_SPEED - abs(reduction)), 'nn'
        except Exception as e:
            logger.warning("Neural net prediction failed: %s — using deterministic fallback", e)
            return 30, 'fallback'

    # If air quality is bad: use OpenAI LLM or configured provider
    if air_quality_index > 50:
        # Pass weather and darkness context to LLM so it can vary its recommendation
        reduction, provenance = get_speed_reduction_from_air_quality(air_quality_index, weather, darkness)
        # Return the final speed and a provenance string (e.g., 'llm_exact', 'fallback_openai').
        return BASE_SPEED - reduction, provenance

    # Otherwise: keep at 80
    return BASE_SPEED, 'base'
