import os
import json
import requests
import pandas as pd
import asyncio
from prompts import Prompts
import re
import hashlib
import time
import datetime
from mistralai import Mistral
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig


PROJECT_ID = "elated-effect-466816-h9"
LOCATION = "us-central1"
vertexai.init(project=PROJECT_ID, location=LOCATION)

model = GenerativeModel("gemini-2.5-flash-lite")
-
def call_gemini(prompt):
    config = GenerationConfig(response_mime_type="application/json")
    try:
        response = model.generate_content(prompt, generation_config=config)
        return response.text.strip()
    except Exception as e:
        return f"ERREUR: {e}"

# Async wrapper for Gemini
async def async_call_gemini(prompt, loop):
    return await loop.run_in_executor(None, call_gemini, prompt)



async def async_call_mistral(prompt, loop):
    return await loop.run_in_executor(None, call_mistral, prompt)


async def clean_questions(question, context, reponse, loop):
    prompt = Prompts.clean_sample(question, context, reponse)

    raw_result = await async_call_gemini(prompt, loop)


    try:
        return json.loads(raw_result)
    except json.JSONDecodeError as e:
        print(f"[JSON ERROR] Parsing failed: {e}")



    def escape_bad_backslashes(s):

        return re.sub(r'(?<!\\)\\(?![\\nt"bfr])', r'\\\\', s)

    try:
        escaped = escape_bad_backslashes(raw_result)
        data = json.loads(escaped)


        if isinstance(data, dict) and "raw_output" in data and isinstance(data["raw_output"], str):
            try:
                nested = json.loads(data["raw_output"])

                for key in ['question', 'context', 'reponse']:
                    if key in nested and isinstance(nested[key], str):
                        nested[key] = nested[key].replace('\\\\', '\\')
                return nested

            except json.JSONDecodeError as e2:
                print(f"[JSON ERROR] Échec parsing de raw_output: {e2}")
                return {"raw_output": data["raw_output"]}

        return data

    except Exception as final_e:
        print(f"[FATAL ERROR] Impossible de parser proprement: {final_e}")
        return {"raw_output": raw_result}
    

def get_safe_filename(subject_path):
    base_name = os.path.splitext(os.path.basename(subject_path))[0]
    base_name = re.sub(r'[<>:"/\\|?*\s]', '_', base_name)
    base_name = re.sub(r'_{2,}', '_', base_name)
    if len(base_name) > 200 or not base_name.strip('_'):
        hash_name = hashlib.md5(subject_path.encode()).hexdigest()[:16]
        base_name = f"file_{hash_name}"
    return base_name


def save_json(data, subject_path, output_dir):
    safe_name = get_safe_filename(subject_path)
    output_path = os.path.join(output_dir, f"{safe_name}.json")

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return output_path
    except OSError as e:
        print(f"Failed to save with sanitized name: {e}")

    try:
        timestamp = int(time.time() * 1000)
        fallback_name = f"processed_{timestamp}.json"
        fallback_path = os.path.join(output_dir, fallback_name)
        with open(fallback_path, "w", encoding="utf-8") as f:
            json.dump({"original_file": subject_path, "data": data}, f, indent=2, ensure_ascii=False)
        print(f"Used timestamp fallback: {fallback_path}")
        return fallback_path
    except OSError as e:
        print(f"Failed with timestamp fallback: {e}")

    try:
        counter = 1
        while True:
            counter_name = f"processed_{counter:04d}.json"
            counter_path = os.path.join(output_dir, counter_name)
            if not os.path.exists(counter_path):
                with open(counter_path, "w", encoding="utf-8") as f:
                    json.dump({"original_file": subject_path, "data": data}, f, indent=2, ensure_ascii=False)
                print(f"Used counter fallback: {counter_path}")
                return counter_path
            counter += 1
            if counter > 10000:
                raise Exception("Could not find available filename")
    except Exception as e:
        print(f"All save strategies failed for {subject_path}: {e}")
        return None


async def progress_callback(message: str):
    now = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] Progress: {message}", flush=True)


async def main(root_pdf, csv_path, output_dir, max_concurrent=10):
    df = pd.read_csv(csv_path)
    loop = asyncio.get_event_loop()
    sem = asyncio.Semaphore(max_concurrent)

    total_files = len(df)
    already_processed = 0

    os.makedirs(output_dir, exist_ok=True)

    processed_files = {os.path.splitext(f)[0] for f in os.listdir(output_dir) if f.endswith(".json")}

    print(f"Checking {total_files} entries...")

    async def process_row(index, row):
        nonlocal already_processed
        async with sem:
            subject_path = os.path.join(root_pdf, "sujets", str(index))
            filename = get_safe_filename(subject_path)

            if filename in processed_files:
                already_processed += 1
                print(f"[SKIPPED] ({already_processed}/{total_files}) Already processed: {subject_path}")
                return

            print(f"[PROCESSING] ({index + 1 - already_processed}/{total_files - already_processed}): {subject_path}")

            try:
                questions = row["questions"]
                context = row["context"]
                responses = row["responses"]
                subject = row["subject"]
                result = await clean_questions(questions, context, responses, loop)
                if result.get("raw_output"):
                    output_path = save_json(result, subject_path, output_dir)
                else:
                    cleaned_entry = {
                        "question": result.get("question", questions),
                        "context": result.get("context", context),
                        "reponse": result.get("reponse", result.get("responses", responses)),
                        "subject": subject
                    }
                    output_path = save_json(cleaned_entry, subject_path, output_dir)

                if output_path:
                    print(f"[SAVED] -> {os.path.basename(output_path)}")
                else:
                    print(f"[ERROR] Failed to save: {subject_path}")

            except Exception as e:
                print(f"[ERROR] Failed to process {subject_path}: {e}")

    tasks = [process_row(i, row) for i, row in df.iterrows()]
    await asyncio.gather(*tasks)

    print("\n=== SUMMARY ===")
    print(f"Total entries: {total_files}")
    print(f"Already processed: {already_processed}")
    print(f"Newly processed: {total_files - already_processed}")

# --- Entrée principale ---
if __name__ == "__main__":
    csv_input = "..."
    output_folder = "..."
    root_pdf = "..." #folder containing the PDF 
    os.makedirs(output_folder, exist_ok=True)
    asyncio.run(main(root_pdf, csv_input, output_folder, max_concurrent=5))

