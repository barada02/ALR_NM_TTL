import random
from pathlib import Path
from shared.config import AGENTIC_PROMPT
from shared.topics import AGENTIC_DOMAINS, AGENTIC_CONSTRAINTS
from shared.generator_base import run_state_machine, create_request_obj

DATASET_TYPE = "agentic"
BASE_DIR = Path(__file__).parent

def build_requests(batch_size: int) -> list[dict]:
    requests = []
    for _ in range(batch_size):
        domain = random.choice(AGENTIC_DOMAINS)
        constraint = random.choice(AGENTIC_CONSTRAINTS)
        prompt = AGENTIC_PROMPT.format(domain=domain, constraint=constraint)
        requests.append(create_request_obj(DATASET_TYPE, prompt))
    return requests

if __name__ == "__main__":
    run_state_machine(DATASET_TYPE, BASE_DIR, build_requests)
