import random
from pathlib import Path
from shared.config import AGENTIC_PROMPT
from shared.topics import AGENTIC_DOMAINS, AGENTIC_ENVIRONMENTS, AGENTIC_TOOLS, AGENTIC_CONSTRAINTS
from shared.generator_base import run_state_machine, create_request_obj

DATASET_TYPE = "agentic"
BASE_DIR = Path(__file__).parent

def build_requests(batch_size: int) -> list[dict]:
    requests = []
    for _ in range(batch_size):
        domain = random.choice(AGENTIC_DOMAINS)
        environment = random.choice(AGENTIC_ENVIRONMENTS)
        tools = random.choice(AGENTIC_TOOLS)
        constraint = random.choice(AGENTIC_CONSTRAINTS)
        prompt = AGENTIC_PROMPT.format(
            domain=domain,
            environment=environment,
            tools=tools,
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
