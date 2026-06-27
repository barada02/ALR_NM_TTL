import os
import json
import random
import uuid
from pathlib import Path

from shared.config import (
    MODEL_NAME, BATCH_SIZE, DATAFORGE_SYSTEM_INSTRUCTION, 
    DIALOGUE_PROMPT, DIALOGUE_TOPICS
)
from shared.batch_utils import BatchManager, StateManager
from shared.extract_utils import Extractor

DATASET_TYPE = "dialogue"
BASE_DIR = Path(__file__).parent
STATE_FILE = BASE_DIR / "state" / f"{DATASET_TYPE}_state.json"
REQUESTS_FILE = BASE_DIR / "state" / f"{DATASET_TYPE}_requests.jsonl"
RAW_RESULTS_FILE = BASE_DIR / "raw_results" / f"{DATASET_TYPE}_raw.jsonl"
PROCESSED_FILE = BASE_DIR / "processed" / f"minititan_{DATASET_TYPE}.jsonl"
DEDUP_DB = BASE_DIR / "state" / "dedup_db.json"

def build_requests():
    """Generates the jsonl file for the Batch API."""
    print(f"Building {BATCH_SIZE} requests for {DATASET_TYPE}...")
    REQUESTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(REQUESTS_FILE, 'w') as f:
        for i in range(BATCH_SIZE):
            topic = random.choice(DIALOGUE_TOPICS)
            prompt = DIALOGUE_PROMPT.format(topic=topic)
            
            request_obj = {
                "key": f"{DATASET_TYPE}_{uuid.uuid4().hex[:8]}",
                "request": {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "system_instruction": {"parts": [{"text": DATAFORGE_SYSTEM_INSTRUCTION}]},
                    "generation_config": {"temperature": 0.9, "max_output_tokens": 2048}
                }
            }
            f.write(json.dumps(request_obj) + "\n")
    print(f"Requests saved to {REQUESTS_FILE}")

def main():
    print(f"--- MiniTitan Data Generation: {DATASET_TYPE.upper()} ---")
    
    state_mgr = StateManager(STATE_FILE)
    status = state_mgr.state.get("status")
    
    print(f"Current State: {status}")
    
    if status == "NOT_STARTED":
        build_requests()
        batch_mgr = BatchManager()
        job_info = batch_mgr.submit_batch(REQUESTS_FILE, f"minititan-{DATASET_TYPE}-batch")
        
        state_mgr.save_state({
            "status": "SUBMITTED",
            "job_name": job_info["job_name"],
            "file_name": job_info["file_name"]
        })
        print("Job submitted successfully. Run this script again later to check status.")
        
    elif status == "SUBMITTED":
        job_name = state_mgr.state.get("job_name")
        if not job_name:
            print("Error: SUBMITTED state but no job_name found.")
            return
            
        batch_mgr = BatchManager()
        job_state, result_file_name = batch_mgr.check_status(job_name)
        
        if "SUCCEEDED" in job_state:
            if result_file_name:
                RAW_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
                batch_mgr.download_results(result_file_name, RAW_RESULTS_FILE)
                
                state_mgr.save_state({"status": "DOWNLOADED"})
                print("Results downloaded. Run this script again to extract the dataset.")
            else:
                print("Job succeeded but could not determine output file name.")
        elif "FAILED" in job_state or "CANCELLED" in job_state:
            print("Job failed or was cancelled. Resetting state.")
            state_mgr.save_state({"status": "NOT_STARTED"})
        else:
            print("Job is still running. Try again later.")
            
    elif status == "DOWNLOADED":
        extractor = Extractor(DEDUP_DB)
        extractor.process_batch_results(RAW_RESULTS_FILE, PROCESSED_FILE, DATASET_TYPE)
        
        state_mgr.save_state({"status": "COMPLETED"})
        print(f"Dataset successfully built at {PROCESSED_FILE}")
        
    elif status == "COMPLETED":
        print(f"Phase complete for {DATASET_TYPE}. Dataset is at {PROCESSED_FILE}.")
        print("To generate the next phase, archive the processed file and delete the state file.")

if __name__ == "__main__":
    main()
