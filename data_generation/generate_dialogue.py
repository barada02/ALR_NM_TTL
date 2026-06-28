import random
from pathlib import Path
from shared.config import DIALOGUE_PROMPT
from shared.topics import DIALOGUE_TOPICS, DIALOGUE_SETTINGS, DIALOGUE_PERSONAS, DIALOGUE_CONSTRAINTS
from shared.generator_base import run_state_machine, create_request_obj

DATASET_TYPE = "dialogue"
BASE_DIR = Path(__file__).parent

def build_requests(batch_size: int) -> list[dict]:
    requests = []
    for _ in range(batch_size):
        topic = random.choice(DIALOGUE_TOPICS)
        setting = random.choice(DIALOGUE_SETTINGS)
        persona = random.choice(DIALOGUE_PERSONAS)
        constraint = random.choice(DIALOGUE_CONSTRAINTS)
        prompt = DIALOGUE_PROMPT.format(
            topic=topic,
            setting=setting,
            persona=persona,
            constraint=constraint
        )
        requests.append(create_request_obj(DATASET_TYPE, prompt))
    return requests

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
    API_KEY = os.environ.get("GEMINI_API_KEY")
    run_state_machine(DATASET_TYPE, BASE_DIR, build_requests, API_KEY)
