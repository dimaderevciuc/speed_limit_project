from openai import OpenAI
import os
import openai
import re
import logging

logger = logging.getLogger(__name__)


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


def get_speed_reduction_from_air_quality(air_quality_index: int):
    """Query the OpenAI Responses API for a suggested reduction.

    If the API call fails (rate limit, quota, network), fall back to a
    deterministic rule-based reduction so the system remains functional.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = f"""
    You are part of a highway control system.
    Based on the current air quality index (AQI = {air_quality_index}), decide by how much
    the speed limit should be reduced in km/h.
    
    The only response you should give is a single integer number indicating the reduction in km/h.
    Do not provide any explanations or additional text.
    
    Rules:
    - If air quality is good (AQI < 50), do not reduce speed.
    - If moderate (50–100), reduce by around 10 km/h.
    - If unhealthy (100–150), reduce by around 20 km/h.
    - If very unhealthy (150–200), reduce by around 30 km/h.
    - If hazardous (>200), reduce by around 40 km/h.
    Respond with only one integer (the reduction in km/h).
    """

    # Improve chance of getting an exact-integer response by:
    #  - setting temperature=0 (deterministic sampling)
    #  - providing tiny few-shot examples in the prompt
    #  - retrying a couple times if the model doesn't return a clean integer
    MAX_ATTEMPTS = 3
    attempt_prompt = (
        """
        Examples (exact outputs shown):
        AQI=40 -> 0
        AQI=120 -> 20
        AQI=175 -> 30
        ---
        """
        + prompt
    )

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.responses.create(
                model="gpt-4o-mini",
                input=attempt_prompt,
                temperature=0,
                max_output_tokens=16,
            )

            try:
                reduction_str = response.output[0].content[0].text.strip()

                # STRICT MODE: accept only exact integer responses (no extra text).
                if re.fullmatch(r"[+-]?\d+", reduction_str):
                    return int(reduction_str), True

                # Not exact — log and retry (unless last attempt)
                logger.warning(
                    "Attempt %d: LLM response not exact integer: %r",
                    attempt,
                    reduction_str,
                )

            except Exception as e:
                logger.warning("Attempt %d: error parsing response: %s", attempt, e)

        except Exception as e:
            # API error (rate limit, quota, network). Log and retry if attempts remain.
            logger.warning("Attempt %d: OpenAI API error: %s", attempt, e)

    # After retries, fall back
    return _deterministic_reduction(air_quality_index), False
