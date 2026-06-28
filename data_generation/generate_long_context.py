import random
from pathlib import Path
from shared.config import LONG_CONTEXT_PROMPT
from shared.topics import LONG_CONTEXT_TOPICS, LONG_CONTEXT_SETTINGS, LONG_CONTEXT_ROLES, LONG_CONTEXT_CONSTRAINTS
from shared.generator_base import run_state_machine, create_request_obj

DATASET_TYPE = "long_context"
BASE_DIR = Path(__file__).parent

def build_requests(batch_size: int) -> list[dict]:
    requests = []
    for _ in range(batch_size):
        topic = random.choice(LONG_CONTEXT_TOPICS)
        setting = random.choice(LONG_CONTEXT_SETTINGS)
        role = random.choice(LONG_CONTEXT_ROLES)
        constraint = random.choice(LONG_CONTEXT_CONSTRAINTS)
        prompt = LONG_CONTEXT_PROMPT.format(
            topic=topic,
            setting=setting,
            role=role,
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
