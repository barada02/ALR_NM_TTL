import random
from pathlib import Path
from shared.config import DIALOGUE_PROMPT
from shared.topics import DIALOGUE_TOPICS
from shared.generator_base import run_state_machine, create_request_obj

DATASET_TYPE = "dialogue"
BASE_DIR = Path(__file__).parent

def build_requests(batch_size: int) -> list[dict]:
    requests = []
    for _ in range(batch_size):
        topic = random.choice(DIALOGUE_TOPICS)
        prompt = DIALOGUE_PROMPT.format(topic=topic)
        requests.append(create_request_obj(DATASET_TYPE, prompt))
    return requests

if __name__ == "__main__":
    run_state_machine(DATASET_TYPE, BASE_DIR, build_requests)
