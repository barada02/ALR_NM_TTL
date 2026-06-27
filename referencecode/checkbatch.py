import os
import sys
import json
import urllib.request
from dotenv import load_dotenv
from google import genai

def main():
    job_name = None
    if len(sys.argv) >= 2:
        job_name = sys.argv[1]
    else:
        # Try to read from the sft job info file
        job_info_path = os.path.join(os.path.dirname(__file__), 'sft_batch_job_info.json')
        if os.path.exists(job_info_path):
            try:
                with open(job_info_path, 'r', encoding='utf-8') as f:
                    job_info = json.load(f)
                    job_name = job_info.get("job_name")
            except Exception as e:
                print(f"[WARNING] Could not read {job_info_path}: {e}")
        
    if not job_name:
        print("Usage: python check_google_batch.py <job_name>")
        print("Example: python check_google_batch.py batches/hu7jauh5l8p89xsnfhinnn1yevz6115tmwoe")
        sys.exit(1)
        
    print(f"--- Checking Status for: {job_name} ---")
    
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("[ERROR] Missing GOOGLE_API_KEY in environment variables.")
        return

    client = genai.Client(api_key=api_key)
    
    try:
        job = client.batches.get(name=job_name)
        print("--- JOB METADATA ---")
        print(f"Display Name: {getattr(job, 'display_name', 'Unknown')}")
        print(f"State:        {job.state}")
        if hasattr(job, 'create_time'):
            print(f"Created At:   {job.create_time}")
        if hasattr(job, 'start_time'):
            print(f"Started At:   {job.start_time}")
        if hasattr(job, 'end_time'):
            print(f"Ended At:     {job.end_time}")
            
        if job.state == "FAILED":
            print(f"!!! ERROR: {job.error if hasattr(job, 'error') else 'Unknown Error'}")
            
        # Try to find counts in 'state_counts' (Standard 2026 structure)
        if hasattr(job, 'state_counts') and job.state_counts:
            completed = getattr(job.state_counts, 'succeeded', 0)
            failed    = getattr(job.state_counts, 'failed', 0)
            total     = getattr(job.state_counts, 'total', 0)
        else:
            # Fallback to the root attributes just in case
            completed = getattr(job, 'completed_request_count', 0)
            failed    = getattr(job, 'failed_request_count', 0)
            total     = getattr(job, 'request_count', 0)

        print(f"Progress: {completed} completed, {failed} failed out of {total} total requests.")

        if "SUCCEEDED" in str(job.state):
            print("\nJob Succeeded!")
            
            # Extract Result File ID (Checking all possible locations)
            file_name = None
            
            # Try 'dest'
            if hasattr(job, 'dest') and job.dest:
                file_name = getattr(job.dest, 'file_name', job.dest)
            # Try 'output'
            elif hasattr(job, 'output') and job.output:
                file_name = getattr(job.output, 'file_name', job.output)
            # Try 'output_config'
            elif hasattr(job, 'output_config') and hasattr(job.output_config, 'file_name'):
                 file_name = job.output_config.file_name
                 
            # MANUAL FALLBACK: If discovery fails, calculate from Job ID
            if not file_name or not isinstance(file_name, str):
                job_id_str = job_name.split('/')[-1]
                file_name = f"files/batch-{job_id_str}"
                print(f"⚠️ Discovery failed, trying manual fallback ID: {file_name}")
                
            print(f"Final Target File: {file_name}")
            
            if file_name:
                print("\nDownloading results via Files API...")
                try:
                    # Fetching as a file object
                    response_bytes = client.files.download(file=file_name)
                    
                    out_file = "sft_batch_results.jsonl"
                    with open(out_file, "wb") as f:
                        f.write(response_bytes)
                    print(f"Successfully downloaded results to: {out_file}")
                except Exception as e:
                    print(f"[ERROR] Failed to download using the Files API: {e}")
                    
        elif "FAILED" in str(job.state):
            print("\nJob Failed! Check Google Cloud Console / AI Studio for details.")
        elif "CANCELLED" in str(job.state):
            print("\nJob was Cancelled.")
        else:
            print("\nJob is still running. Try checking again later.")
            
    except Exception as e:
        print(f"[ERROR] Failed to get job status: {e}")

if __name__ == "__main__":
    main()