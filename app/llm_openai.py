from openai import OpenAI
import os
import openai
import re
import logging
import json
import time
import random
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)

# Support multiple LLM providers via env var LLM_PROVIDER:
# - 'openai' (default): use OpenAI Responses API
# - 'local': use a local transformers model (flan-t5-small by default)
# - 'mock': return deterministic or simulated responses for testing
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
USE_LLM = os.getenv("USE_LLM", "true").lower() != "false"
MOCK_LLM = os.getenv("MOCK_LLM", "false").lower() == "true"
LLM_CACHE = os.getenv("LLM_CACHE", "true").lower() != "false"
try:
    LLM_CACHE_TTL = int(os.getenv("LLM_CACHE_TTL", "3600"))
except Exception:
    LLM_CACHE_TTL = 3600

# Lazy-loaded local model variables
_local_tokenizer = None
_local_model = None


def _load_local_model(model_name: str = "google/flan-t5-small"):
    """Lazy-load transformers tokenizer and model onto CPU."""
    global _local_tokenizer, _local_model
    if _local_model is not None and _local_tokenizer is not None:
        return _local_tokenizer, _local_model

    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    except Exception as e:
        raise RuntimeError("transformers is required for local LLM provider: pip install transformers") from e

    _local_tokenizer = AutoTokenizer.from_pretrained(model_name)
    _local_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return _local_tokenizer, _local_model


# Simple file-backed cache helpers
_CACHE_PATH = Path(__file__).resolve().parents[1] / ".cache"
_CACHE_FILE = _CACHE_PATH / "aqi_reduction_cache.json"

def _ensure_cache_dir():
    try:
        _CACHE_PATH.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

def _read_cache(key: str):
    if not LLM_CACHE:
        return None
    try:
        if not _CACHE_FILE.exists():
            return None
        with _CACHE_FILE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        entry = data.get(key)
        if not entry:
            return None
        # TTL check
        if int(time.time()) - int(entry.get("ts", 0)) > LLM_CACHE_TTL:
            return None
        return entry
    except Exception:
        return None

def _write_cache(key: str, value, prov: str = "cache"):
    if not LLM_CACHE:
        return
    try:
        _ensure_cache_dir()
        data = {}
        if _CACHE_FILE.exists():
            try:
                with _CACHE_FILE.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception:
                data = {}
        data[key] = {"value": value, "prov": prov, "ts": int(time.time())}
        with _CACHE_FILE.open("w", encoding="utf-8") as fh:
            json.dump(data, fh)
    except Exception:
        pass


def _deterministic_reduction(aqi: int) -> int:
    if aqi < 50:
        return 0
    if aqi <= 100:
        return 10
    if aqi <= 150:
        return 20
    if aqi <= 200:
        return 30
    return 40


def get_speed_reduction_from_air_quality(air_quality_index: int, weather: str = "good", darkness: bool = False):
    """Query the OpenAI Responses API for a suggested reduction.

    If the API call fails (rate limit, quota, network), fall back to a
    deterministic rule-based reduction so the system remains functional.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = f"""
    You are part of a highway control system. Decide how many km/h the speed limit should be reduced.
    Provide a single integer output only (no text, units, or explanation).

    Context (the caller will supply these values):
    - AQI: {air_quality_index}
    - Weather: {{weather}}    # one of: good, bad
    - Darkness: {{darkness}}  # True or False

    IMPORTANT: Your output MUST be exactly one integer token (for example: 10). Do NOT include any words,
    units, punctuation, or explanation. Only output the number. If you cannot, output nothing.
    """

    # Compute a safe deterministic base reduction, then ask the LLM for a
    # small adjustment. This keeps outputs safe while allowing contextual
    # variation from the model.
    base_reduction = _deterministic_reduction(air_quality_index)

    # Cache key for this query (based on input and base reduction)
    cache_key = f"{air_quality_index}|{weather}|{darkness}|{base_reduction}"
    cached = _read_cache(cache_key)
    if cached is not None:
        logger.debug("Cache hit for %s -> %s", cache_key, cached)
        return cached.get("value"), cached.get("prov", "cache")

    # If LLM use is disabled or mock mode is requested, handle accordingly
    if not USE_LLM:
        return base_reduction, "fallback_disabled"

    if MOCK_LLM:
        # Simple mock that follows deterministic mapping but marks as 'llm' for testing
        return _deterministic_reduction(air_quality_index), "llm_mock"

    # Read temperature from env (default 0.2 to allow slight, sensible variation)
    try:
        LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    except Exception:
        LLM_TEMPERATURE = 0.2

    # Provider selection
    provider = LLM_PROVIDER
    if provider == "local":
        # Use local transformers model
        try:
            tokenizer, model = _load_local_model(os.getenv("LOCAL_LLM_MODEL", "google/flan-t5-small"))
        except Exception as e:
            logger.warning("Local model load failed: %s", e)
            return _deterministic_reduction(air_quality_index), False

        # Few-shot examples and strict instruction: model MUST output exactly one
        # integer token (an adjustment in km/h), nothing else. If it cannot,
        # respond with nothing or a non-integer (which we will treat as failure).
        examples = (
            "Instruction: Given AQI,Weather,Darkness,BaseReduction output EXACTLY one integer adjustment in km/h (between -5 and 5).\n"
            "Do NOT output any words, punctuation, units or explanation — only the integer.\n"
            "Examples:\n"
            "AQI=40,Weather=good,Darkness=False,Base=0 => 0\n"
            "AQI=55,Weather=good,Darkness=False,Base=10 => 0\n"
            "AQI=80,Weather=good,Darkness=True,Base=15 => 2\n"
            "AQI=90,Weather=bad,Darkness=False,Base=18 => 3\n"
            "AQI=120,Weather=bad,Darkness=False,Base=25 => 0\n"
            "AQI=175,Weather=bad,Darkness=True,Base=35 => 2\n"
            "Now respond for the following input (output must be a single integer):\n"
        )

        weather_str = str(weather)
        darkness_str = str(darkness)
        prompt_local = examples + f"AQI={air_quality_index},Weather={weather_str},Darkness={darkness_str},Base={base_reduction} =>"

        # Do a strict generation pass: greedy/beam decoding with low max tokens
        try:
            inputs = tokenizer(prompt_local, return_tensors="pt")
            # Use deterministic decoding (no sampling) to encourage exact tokens
            out = model.generate(**inputs, max_new_tokens=3, do_sample=False, num_beams=3, early_stopping=True)
            text = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        except Exception as e:
            logger.warning("Local LLM generation failed: %s", e)
            _write_cache(cache_key, base_reduction)
            return base_reduction, False

        # Accept only exact integer token outputs. If not exact, treat as failure.
        m_exact = re.fullmatch(r"[+-]?\d+", text)
        if m_exact:
            adj = int(m_exact.group())
            final = max(0, base_reduction + adj)
            _write_cache(cache_key, final, "llm_exact")
            return final, "llm_exact"

        logger.warning("Local LLM did not return an exact integer: %r", text)
        _write_cache(cache_key, base_reduction, "fallback_local")
        return base_reduction, "fallback_local"

    # Default: use OpenAI provider (existing logic, deterministic sampling + retries)
    MAX_ATTEMPTS = 3
    # Compact few-shot examples for OpenAI provider. We include the BaseReduction
    # field and ask the model for a small integer adjustment to that base.
    examples = (
        "Instruction: Given AQI,Weather,Darkness,BaseReduction return a single integer adjustment in km/h (between -5 and 5).\n"
        "Examples:\n"
        "AQI=40,Weather=good,Darkness=False,Base=0 => 0\n"
        "AQI=55,Weather=good,Darkness=False,Base=10 => 0\n"
        "AQI=80,Weather=good,Darkness=True,Base=15 => 2\n"
        "AQI=90,Weather=bad,Darkness=False,Base=18 => 3\n"
        "AQI=120,Weather=bad,Darkness=False,Base=25 => 0\n"
        "AQI=175,Weather=bad,Darkness=True,Base=35 => 2\n"
        "Now respond for the following input:\n"
    )

    attempt_prompt = examples + f"AQI={air_quality_index},Weather={weather},Darkness={darkness},Base={base_reduction} =>"
    # Check cache for OpenAI provider as well
    cached = _read_cache(cache_key)
    if cached is not None:
        logger.debug("Cache hit for %s -> %s", cache_key, cached)
        return cached, False

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.responses.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                input=attempt_prompt,
                temperature=0,
                max_output_tokens=16,
            )

            try:
                reduction_str = response.output[0].content[0].text.strip()

                # Strict mode: accept only exact integer adjustment tokens. If the
                # model does not return an exact integer, we treat the LLM as
                # unsuccessful and fall back to the deterministic base.
                m_exact = re.fullmatch(r"[+-]?\d+", reduction_str)
                if m_exact:
                    adj = int(m_exact.group())
                    final = max(0, base_reduction + adj)
                    _write_cache(cache_key, final, "llm_exact")
                    return final, "llm_exact"

                logger.warning("OpenAI response not an exact integer: %r", reduction_str)

            except Exception as e:
                logger.warning("Attempt %d: error parsing response: %s", attempt, e)

        except Exception as e:
            # API error (rate limit, quota, network). Log and retry if attempts remain.
            logger.warning("Attempt %d: OpenAI API error: %s", attempt, e)

    # After retries, fall back to the previously computed base reduction
    _write_cache(cache_key, base_reduction, "fallback_openai")
    return base_reduction, "fallback_openai"
