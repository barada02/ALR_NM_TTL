import random
from pathlib import Path
from shared.config import LONG_CONTEXT_PROMPT
from shared.topics import LONG_CONTEXT_TOPICS, LONG_CONTEXT_CONSTRAINTS
from shared.generator_base import run_state_machine, create_request_obj

DATASET_TYPE = "long_context"
BASE_DIR = Path(__file__).parent

def build_requests(batch_size: int) -> list[dict]:
    requests = []
    for _ in range(batch_size):
        topic = random.choice(LONG_CONTEXT_TOPICS)
        constraint = random.choice(LONG_CONTEXT_CONSTRAINTS)
        prompt = LONG_CONTEXT_PROMPT.format(topic=topic, constraint=constraint)
        requests.append(create_request_obj(DATASET_TYPE, prompt))
    return requests

if __name__ == "__main__":
    run_state_machine(DATASET_TYPE, BASE_DIR, build_requests)
