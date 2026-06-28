import os
import random
from pathlib import Path
from shared.config import LONG_CONTEXT_PROMPT, DIALOGUE_PROMPT, AGENTIC_PROMPT
from shared.topics import (
    LONG_CONTEXT_TOPICS, LONG_CONTEXT_SETTINGS, LONG_CONTEXT_ROLES, LONG_CONTEXT_CONSTRAINTS,
    DIALOGUE_TOPICS, DIALOGUE_SETTINGS, DIALOGUE_PERSONAS, DIALOGUE_CONSTRAINTS,
    AGENTIC_DOMAINS, AGENTIC_ENVIRONMENTS, AGENTIC_TOOLS, AGENTIC_CONSTRAINTS
)
from shared.generator_base import run_state_machine, create_request_obj

DATASET_TYPE = "test_batch"
BASE_DIR = Path(__file__).parent

def build_test_requests(batch_size: int) -> list[dict]:
    # We ignore the config batch_size and force it to 6 requests for testing
    requests = []
    
    # 1. Long Context (2 requests)
    for _ in range(2):
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
        requests.append(create_request_obj(f"{DATASET_TYPE}_lc", prompt))
        
    # 2. Dialogue (2 requests)
    for _ in range(2):
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
        requests.append(create_request_obj(f"{DATASET_TYPE}_diag", prompt))
        
    # 3. Agentic (2 requests)
    for _ in range(2):
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
        requests.append(create_request_obj(f"{DATASET_TYPE}_agent", prompt))
        
    return requests

from dotenv import load_dotenv

# Load env variables at the top level
load_dotenv(Path(__file__).parent / ".env")
API_KEY = os.environ.get("GEMINI_API_KEY")

if __name__ == "__main__":
    # This will reuse all the robust state machine logic, but with our test batch of 6!
    run_state_machine(DATASET_TYPE, BASE_DIR, build_test_requests, API_KEY)
