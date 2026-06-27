import os
from pathlib import Path
from shared.config import LONG_CONTEXT_PROMPT, DIALOGUE_PROMPT, AGENTIC_PROMPT
from shared.topics import LONG_CONTEXT_TOPICS, DIALOGUE_TOPICS, AGENTIC_DOMAINS
from shared.generator_base import run_state_machine, create_request_obj

DATASET_TYPE = "test_batch"
BASE_DIR = Path(__file__).parent

def build_test_requests(batch_size: int) -> list[dict]:
    # We ignore the config batch_size and force it to 6 requests for testing
    requests = []
    
    # 1. Long Context (2 requests)
    topic_lc = LONG_CONTEXT_TOPICS[0]
    for _ in range(2):
        prompt = LONG_CONTEXT_PROMPT.format(topic=topic_lc)
        requests.append(create_request_obj(f"{DATASET_TYPE}_lc", prompt))
        
    # 2. Dialogue (2 requests)
    topic_dialogue = DIALOGUE_TOPICS[0]
    for _ in range(2):
        prompt = DIALOGUE_PROMPT.format(topic=topic_dialogue)
        requests.append(create_request_obj(f"{DATASET_TYPE}_diag", prompt))
        
    # 3. Agentic (2 requests)
    domain_agentic = AGENTIC_DOMAINS[0]
    for _ in range(2):
        prompt = AGENTIC_PROMPT.format(domain=domain_agentic)
        requests.append(create_request_obj(f"{DATASET_TYPE}_agent", prompt))
        
    return requests

import os
from dotenv import load_dotenv

# Load env variables at the top level
load_dotenv(Path(__file__).parent / ".env")
API_KEY = os.environ.get("GEMINI_API_KEY")

if __name__ == "__main__":
    # This will reuse all the robust state machine logic, but with our test batch of 6!
    run_state_machine(DATASET_TYPE, BASE_DIR, build_test_requests, API_KEY)
