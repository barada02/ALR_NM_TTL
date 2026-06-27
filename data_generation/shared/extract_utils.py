import json
import hashlib
from pathlib import Path

class Extractor:
    def __init__(self, dedup_db_path: Path):
        self.dedup_db_path = dedup_db_path
        self.seen_hashes = self.load_dedup_db()
        
    def load_dedup_db(self) -> set:
        if self.dedup_db_path.exists():
            with open(self.dedup_db_path, 'r') as f:
                try:
                    data = json.load(f)
                    return set(data.get("hashes", []))
                except json.JSONDecodeError:
                    return set()
        return set()

    def save_dedup_db(self):
        self.dedup_db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.dedup_db_path, 'w') as f:
            json.dump({"hashes": list(self.seen_hashes)}, f)

    def dedup_key(self, sample: dict) -> str:
        """Generate a deduplication hash for a sample."""
        content = json.dumps(sample, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    def parse_gemini_json(self, text: str) -> dict | None:
        """Safely parse Gemini's JSON response."""
        if not text:
            return None
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if len(lines) > 2:
                text = "\n".join(lines[1:-1])
            else:
                return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def process_batch_results(self, raw_jsonl_path: Path, output_jsonl_path: Path, dataset_type: str):
        """Extracts JSON from batch results, deduplicates, and saves final dataset."""
        print(f"Extracting {dataset_type} from {raw_jsonl_path}...")
        
        output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        
        valid_count = 0
        duplicate_count = 0
        error_count = 0
        
        # We append to output, in case we are accumulating
        with open(raw_jsonl_path, 'r') as infile, open(output_jsonl_path, 'a') as outfile:
            for line in infile:
                try:
                    response_item = json.loads(line)
                    # The response structure from Gemini Batch API usually wraps the output
                    # depending on the schema. We'll look for the text part.
                    # typically: {"response": {"candidates": [{"content": {"parts": [{"text": "..."}]}}]}}
                    
                    if "response" in response_item and "candidates" in response_item["response"]:
                        candidates = response_item["response"]["candidates"]
                        if candidates and "content" in candidates[0]:
                            parts = candidates[0]["content"].get("parts", [])
                            if parts and "text" in parts[0]:
                                raw_text = parts[0]["text"]
                                
                                parsed = self.parse_gemini_json(raw_text)
                                if parsed:
                                    h = self.dedup_key(parsed)
                                    if h in self.seen_hashes:
                                        duplicate_count += 1
                                    else:
                                        self.seen_hashes.add(h)
                                        parsed["_type"] = dataset_type
                                        outfile.write(json.dumps(parsed) + "\n")
                                        valid_count += 1
                                    continue
                    error_count += 1
                except Exception as e:
                    error_count += 1
                    
        self.save_dedup_db()
        print(f"Extraction complete for {dataset_type}: {valid_count} valid, {duplicate_count} duplicates, {error_count} errors.")
