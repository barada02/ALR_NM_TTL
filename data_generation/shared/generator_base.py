import json
import uuid
from pathlib import Path
from typing import Callable

from shared.config import DATAFORGE_SYSTEM_INSTRUCTION, BATCH_SIZE
from shared.batch_utils import BatchManager, StateManager
from shared.extract_utils import Extractor

def run_state_machine(
    dataset_type: str,
    base_dir: Path,
    prompt_builder_fn: Callable[[int], list[dict]],
    api_key: str
):
    STATE_FILE = base_dir / "state" / f"{dataset_type}_state.json"
    REQUESTS_FILE = base_dir / "state" / f"{dataset_type}_requests.jsonl"
    RAW_RESULTS_FILE = base_dir / "raw_results" / f"{dataset_type}_raw.jsonl"
    PROCESSED_FILE = base_dir / "processed" / f"minititan_{dataset_type}.jsonl"
    DEDUP_DB = base_dir / "state" / "dedup_db.json"
    
    print(f"--- MiniTitan Data Generation: {dataset_type.upper()} ---")
    
    state_mgr = StateManager(STATE_FILE)
    status = state_mgr.state.get("status")
    
    print(f"Current State: {status}")
    
    if status == "NOT_STARTED":
        print(f"Building requests for {dataset_type}...")
        REQUESTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        requests = prompt_builder_fn(BATCH_SIZE)
        
        with open(REQUESTS_FILE, 'w') as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")
        print(f"Requests saved to {REQUESTS_FILE}")
        
        batch_mgr = BatchManager()
        job_info = batch_mgr.submit_batch(REQUESTS_FILE, f"minititan-{dataset_type}-batch")
        
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
        batch_mgr = BatchManager(api_key=api_key)
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
        extractor.process_batch_results(RAW_RESULTS_FILE, PROCESSED_FILE, dataset_type)
        
        state_mgr.save_state({"status": "COMPLETED"})
        print(f"Dataset successfully built at {PROCESSED_FILE}")
        
    elif status == "COMPLETED":
        print(f"Phase complete for {dataset_type}. Dataset is at {PROCESSED_FILE}.")
        print("To generate the next phase, archive the processed file and delete the state file.")

def create_request_obj(dataset_type: str, prompt: str) -> dict:
    return {
        "key": f"{dataset_type}_{uuid.uuid4().hex[:8]}",
        "request": {
            "contents": [{"parts": [{"text": prompt}]}],
            "system_instruction": {"parts": [{"text": DATAFORGE_SYSTEM_INSTRUCTION}]},
            "generation_config": {"temperature": 0.9, "max_output_tokens": 2048}
        }
    }
