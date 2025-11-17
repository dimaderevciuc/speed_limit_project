import torch
import numpy as np
from pathlib import Path
from model.train_nn import SpeedLimitNN
from app.llm_openai import get_speed_reduction_from_air_quality


def load_nn_model():
    # Resolve model path relative to project root (not CWD)
    base = Path(__file__).resolve().parents[1]
    model_path = base / 'model' / 'speed_limit_nn.pth'

    model = SpeedLimitNN()
    # Load onto CPU by default for portability
    state = torch.load(str(model_path), map_location=torch.device('cpu'))
    model.load_state_dict(state)
    model.eval()
    return model


def calculate_speed_limit(weather, darkness, air_quality_index):
    BASE_SPEED = 80
    model = load_nn_model()

    # If bad weather or dark: use NN prediction
    if darkness or weather == "bad":
        X = np.array([[80, 60, 25, 110, 10000]], dtype=np.float32)
        with torch.no_grad():
            # Ensure torch tensor dtype and extract scalar with .item()
            inp = torch.tensor(X, dtype=torch.float32)
            out = model(inp)
            reduction = float(out.item())
        return max(30, BASE_SPEED - abs(reduction)), 'nn'

    # If air quality is bad: use OpenAI LLM
    if air_quality_index > 50:
        reduction, used_llm = get_speed_reduction_from_air_quality(air_quality_index)
        source = 'llm' if used_llm else 'fallback'
        return BASE_SPEED - reduction, source

    # Otherwise: keep at 80
    return BASE_SPEED, 'base'
