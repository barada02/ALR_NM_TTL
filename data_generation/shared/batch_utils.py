import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

class BatchManager:
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            api_key = os.environ.get("GOOGLE_API_KEY")
            
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY or GOOGLE_API_KEY in environment variables.")
        self.client = genai.Client(api_key=api_key)

    def submit_batch(self, jsonl_file_path: Path, display_name: str) -> dict:
        """Uploads a jsonl file and creates a batch job. Returns job info."""
        print(f"Uploading file: {jsonl_file_path.name} ...")
        uploaded_file = self.client.files.upload(
            file=str(jsonl_file_path),
            config=types.UploadFileConfig(
                display_name=display_name, 
                mime_type='application/jsonl'
            )
        )
        print(f"Successfully uploaded file! URI: {uploaded_file.name}")
        
        print("Creating batch job...")
        file_batch_job = self.client.batches.create(
            model="gemini-1.5-flash",
            src=uploaded_file.name,
            config={
                'display_name': display_name,
            },
        )
        print(f"Created batch job successfully! Job Name: {file_batch_job.name}")
        
        return {
            "file_name": uploaded_file.name,
            "job_name": file_batch_job.name,
            "job_state": str(file_batch_job.state)
        }

    def check_status(self, job_name: str) -> tuple[str, str | None]:
        """Checks job status. Returns (status, downloaded_file_uri_if_success)."""
        job = self.client.batches.get(name=job_name)
        
        state = str(job.state)
        print(f"Job State: {state}")
        
        # Try to find counts
        if hasattr(job, 'state_counts') and job.state_counts:
            completed = getattr(job.state_counts, 'succeeded', 0)
            failed    = getattr(job.state_counts, 'failed', 0)
            total     = getattr(job.state_counts, 'total', 0)
            print(f"Progress: {completed} completed, {failed} failed out of {total} total requests.")
        
        file_name = None
        if "SUCCEEDED" in state:
            if hasattr(job, 'dest') and job.dest:
                file_name = getattr(job.dest, 'file_name', job.dest)
            elif hasattr(job, 'output') and job.output:
                file_name = getattr(job.output, 'file_name', job.output)
            elif hasattr(job, 'output_config') and hasattr(job.output_config, 'file_name'):
                 file_name = job.output_config.file_name
                 
            if not file_name or not isinstance(file_name, str):
                job_id_str = job_name.split('/')[-1]
                file_name = f"files/batch-{job_id_str}"
                
        return state, file_name

    def download_results(self, file_name: str, output_path: Path):
        """Downloads the results file."""
        print(f"Downloading results via Files API from {file_name}...")
        response_bytes = self.client.files.download(file=file_name)
        with open(output_path, "wb") as f:
            f.write(response_bytes)
        print(f"Successfully downloaded results to: {output_path}")

class StateManager:
    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state = self.load_state()

    def load_state(self) -> dict:
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {"status": "NOT_STARTED"}

    def save_state(self, state_dict: dict):
        self.state.update(state_dict)
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=4)
