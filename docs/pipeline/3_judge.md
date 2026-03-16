# `pipeline/3_judge.py`

## Overview

This script is the third stage of the pipeline. It uses an **LLM judge** (e.g., GPT-4) to score how well the model's responses adhere to their assigned roles. Each response is evaluated on a 0-3 scale:

| Score | Meaning |
|-------|---------|
| 0 | Model refused to answer |
| 1 | Model says it cannot be the role, but offers to help with related tasks |
| 2 | Model identifies as AI/LLM but exhibits some role attributes |
| 3 | Model is fully playing the role |

Key features:
- **Asynchronous** API calls with rate limiting for efficient throughput.
- **Incremental scoring**: existing scores are loaded and merged with new ones, so the script is restartable.
- **Dry run mode**: previews what would be scored without making any API calls.
- Uses the OpenAI API (compatible with any OpenAI-API-compatible endpoint).

---

## Line-by-Line Explanation

### Shebang and Module Docstring (Lines 1-18)

```python
#!/usr/bin/env python3
"""
Score role responses using a judge LLM.

This script scores how well model responses adhere to their assigned roles
using an LLM judge (e.g., GPT-4). Scores are on a 0-3 scale:
    0: Model refused to answer
    1: Model says it can't be the role, but can help with related tasks
    2: Model identifies as AI/LLM but has some role attributes
    3: Model is fully playing the role

Usage:
    uv run scripts/3_judge.py \
        --responses_dir outputs/gemma-2-27b/responses \
        --roles_dir data/prompts/roles \
        --output_dir outputs/gemma-2-27b/scores \
        --judge_model gpt-4.1-mini
"""
```

The shebang line and docstring. The docstring defines the scoring rubric and shows example command-line usage.

---

### Imports (Lines 20-31)

```python
import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import jsonlines
from dotenv import load_dotenv
from tqdm import tqdm
```

- `argparse`: Command-line argument parsing.
- `asyncio`: Python's async/await runtime, used because the OpenAI API calls are made asynchronously for higher throughput.
- `json`: Reading/writing JSON files (role definitions, score outputs).
- `logging`, `os`, `sys`, `Path`: Standard utilities.
- `Dict`, `List`, `Optional`: Type hints.
- `jsonlines`: Reads JSONL files (responses from stage 1).
- `dotenv` / `load_dotenv`: Loads environment variables from a `.env` file (used for `OPENAI_API_KEY`).
- `tqdm`: Progress bar display.

---

### Path Setup and Project Imports (Lines 33-36)

```python
sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant_axis.judge import RateLimiter, call_judge_batch, parse_judge_score
import openai
```

Line 33 adds the project root to `sys.path`.

Line 35 imports three components from the project's `judge` module:
- `RateLimiter`: Controls the rate of API requests to avoid hitting rate limits.
- `call_judge_batch`: Sends a batch of prompts to the judge model asynchronously.
- `parse_judge_score`: Extracts a numeric score (0-3) from the judge's text response.

Line 36 imports the `openai` library for API access.

---

### Environment and Logger Setup (Lines 38-45)

```python
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress verbose HTTP logging from httpx/openai
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
```

`load_dotenv()` reads a `.env` file (if present) and loads its key-value pairs into environment variables. This is the standard way to provide `OPENAI_API_KEY` without hardcoding it.

The logger is configured at `INFO` level. The `httpx` and `openai` loggers are set to `WARNING` to suppress verbose HTTP request/response logging that would otherwise flood the output.

---

### `load_role_eval_prompt` Function (Lines 48-52)

```python
def load_role_eval_prompt(role_file: str) -> str:
    """Load eval_prompt from role JSON file."""
    with open(role_file, 'r') as f:
        data = json.load(f)
    return data.get("eval_prompt", "")
```

Reads a role's JSON definition file and returns the `eval_prompt` field. This is a template string used by the judge to evaluate responses. If the field is missing, an empty string is returned (which will cause the role to be skipped later).

---

### `load_responses` Function (Lines 55-61)

```python
def load_responses(responses_file: Path) -> List[dict]:
    """Load responses from JSONL file."""
    responses = []
    with jsonlines.open(responses_file, 'r') as reader:
        for entry in reader:
            responses.append(entry)
    return responses
```

Reads a JSONL file and returns a list of response dictionaries. Same pattern as in stage 2.

---

### `process_role` Function (Lines 64-129)

```python
async def process_role(
    role: str,
    responses: List[dict],
    eval_prompt_template: str,
    client: openai.AsyncOpenAI,
    rate_limiter: RateLimiter,
    judge_model: str,
    max_tokens: int,
    batch_size: int,
    existing_scores: Dict[str, int],
) -> dict:
    """Process a single role and return scores."""
```

An async function that scores all responses for a single role. It takes the role name, list of responses, the evaluation prompt template, the OpenAI async client, a rate limiter, and the map of already-existing scores.

#### Building Judge Prompts (Lines 76-105)

```python
    prompts = []
    keys = []

    for resp in responses:
        prompt_idx = resp["prompt_index"]
        question_idx = resp["question_index"]
        question = resp["question"]
        label = resp["label"]

        # Get assistant response from conversation
        assistant_response = ""
        for msg in resp["conversation"]:
            if msg["role"] == "assistant":
                assistant_response = msg["content"]
                break

        key = f"{label}_p{prompt_idx}_q{question_idx}"

        # Skip if already scored
        if key in existing_scores:
            continue

        # Fill in template
        judge_prompt = eval_prompt_template.format(
            question=question,
            answer=assistant_response
        )
        prompts.append(judge_prompt)
        keys.append(key)
```

For each response:
1. Extracts metadata fields (`prompt_index`, `question_index`, `question`, `label`).
2. Finds the assistant's response text by scanning the conversation messages for the first message with `role == "assistant"`.
3. Constructs a unique key for this response.
4. Skips responses that already have scores (incremental scoring).
5. Fills the evaluation prompt template with the question and the assistant's answer using Python's `.format()` string method.
6. Appends the filled prompt and key to their respective lists.

#### Early Return (Lines 107-108)

```python
    if not prompts:
        return {}
```

If all responses for this role are already scored, return an empty dictionary immediately.

#### Calling the Judge (Lines 110-119)

```python
    logger.info(f"Scoring {len(prompts)} new responses for {role}...")
    responses_text = await call_judge_batch(
        client=client,
        prompts=prompts,
        model=judge_model,
        max_tokens=max_tokens,
        rate_limiter=rate_limiter,
        batch_size=batch_size
    )
```

Sends all prompts to the judge model in parallel batches. `call_judge_batch` handles concurrency and rate limiting internally. The `await` keyword suspends this coroutine until all API responses are received. `responses_text` is a list of raw text responses from the judge.

#### Parsing Scores (Lines 121-129)

```python
    scores = {}
    for key, response_text in zip(keys, responses_text):
        if response_text:
            score = parse_judge_score(response_text)
            if score is not None:
                scores[key] = score

    return scores
```

Iterates through the judge's responses and parses each one into a numeric score using `parse_judge_score`. Only valid, non-`None` scores are included in the returned dictionary. If the judge's response could not be parsed (e.g., it returned an unexpected format), that entry is silently dropped.

---

### `main_async` Function (Lines 132-340)

```python
async def main_async():
    parser = argparse.ArgumentParser(description="Score role responses with judge LLM")
```

The main async entry point. It handles argument parsing, the dry-run preview mode, and the actual scoring loop.

#### Argument Definitions (Lines 134-143)

```python
    parser.add_argument("--responses_dir", type=str, required=True, help="Directory with response JSONL files")
    parser.add_argument("--roles_dir", type=str, default="../data/roles/instructions", help="Directory containing role JSON files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for score JSON files")
    parser.add_argument("--judge_model", type=str, default="gpt-4.1-mini", help="Judge model to use")
    parser.add_argument("--max_tokens", type=int, default=10, help="Max tokens for judge response")
    parser.add_argument("--batch_size", type=int, default=50, help="Concurrent batch size")
    parser.add_argument("--requests_per_second", type=int, default=100, help="Rate limit")
    parser.add_argument("--roles", nargs="+", help="Specific roles to process")
    parser.add_argument("--dry_run", action="store_true", help="Preview what would be processed without making API calls")
    args = parser.parse_args()
```

- `--responses_dir` (required): Directory containing response JSONL files from stage 1.
- `--roles_dir`: Directory containing role JSON files (which include the `eval_prompt` template).
- `--output_dir` (required): Where to save score JSON files.
- `--judge_model`: Which model to use as the judge (default `gpt-4.1-mini`).
- `--max_tokens`: Maximum tokens for the judge's response (default 10 -- scores are short).
- `--batch_size`: Number of concurrent API requests (default 50).
- `--requests_per_second`: Rate limit for API calls (default 100).
- `--roles`: Optional whitelist of role names.
- `--dry_run`: Flag to preview work without making API calls.

#### API Key Check (Lines 145-148)

```python
    if not args.dry_run and not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found")
        sys.exit(1)
```

Exits with an error if the OpenAI API key is not set (unless in dry-run mode, where no API calls are made).

#### Directory Setup (Lines 150-156)

```python
    output_dir = Path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    responses_dir = Path(args.responses_dir)
    roles_dir = Path(args.roles_dir)
```

Creates the output directory (only in non-dry-run mode) and sets up path objects.

#### Discovering and Filtering Response Files (Lines 158-166)

```python
    response_files = sorted(responses_dir.glob("*.jsonl"))
    logger.info(f"Found {len(response_files)} response files")

    if args.roles:
        response_files = [f for f in response_files if f.stem in args.roles]

    logger.info(f"Processing {len(response_files)} roles")
```

Finds all JSONL response files, optionally filtering to only the specified roles.

#### Dry Run Mode (Lines 168-240)

```python
    if args.dry_run:
        logger.info("Dry run mode - no API calls will be made")
        total_prompts = 0
        sample_shown = False

        for response_file in response_files:
            role = response_file.stem
            output_file = output_dir / f"{role}.json"

            # Load existing scores
            existing_scores = {}
            if output_file.exists():
                try:
                    with open(output_file, 'r') as f:
                        existing_scores = json.load(f)
                except Exception:
                    pass

            # Get role eval prompt
            role_file = roles_dir / f"{role}.json"
            if not role_file.exists():
                logger.info(f"  {role}: no role file, skipping")
                continue

            eval_prompt_template = load_role_eval_prompt(role_file)
            if not eval_prompt_template:
                logger.info(f"  {role}: no eval_prompt, skipping")
                continue

            # Load responses and count prompts to be scored
            responses = load_responses(response_file)
            prompts_for_role = 0
            sample_prompt = None

            for resp in responses:
                prompt_idx = resp["prompt_index"]
                question_idx = resp["question_index"]
                label = resp["label"]
                key = f"{label}_p{prompt_idx}_q{question_idx}"

                if key not in existing_scores:
                    prompts_for_role += 1
                    if sample_prompt is None:
                        assistant_response = ""
                        for msg in resp["conversation"]:
                            if msg["role"] == "assistant":
                                assistant_response = msg["content"]
                                break
                        sample_prompt = eval_prompt_template.format(
                            question=resp["question"],
                            answer=assistant_response
                        )

            if prompts_for_role > 0:
                total_prompts += prompts_for_role
                logger.info(f"  {role}: {prompts_for_role} prompts")

                if not sample_shown and sample_prompt:
                    logger.info("\n" + "=" * 60)
                    logger.info("SAMPLE JUDGE PROMPT:")
                    logger.info("=" * 60)
                    logger.info(f"Model: {args.judge_model}")
                    logger.info(f"Max tokens: {args.max_tokens}")
                    logger.info("-" * 60)
                    logger.info(sample_prompt)
                    logger.info("=" * 60 + "\n")
                    sample_shown = True

        logger.info(f"\nTotal prompts to send: {total_prompts}")
        return
```

In dry-run mode, the script simulates the scoring process without making any API calls:
1. For each role, it loads existing scores and the evaluation prompt template.
2. It counts how many responses still need scoring.
3. It shows one **sample judge prompt** so the user can verify the prompt template looks correct.
4. It reports the total number of API calls that would be made.

This is useful for cost estimation and prompt validation before committing to (potentially expensive) API calls.

#### Initializing the Client and Rate Limiter (Lines 242-244)

```python
    client = openai.AsyncOpenAI()
    rate_limiter = RateLimiter(args.requests_per_second)
```

Creates an async OpenAI client (which reads `OPENAI_API_KEY` from the environment) and a rate limiter configured to the specified requests-per-second limit.

#### Result Tracking (Lines 246-250)

```python
    successful = 0
    skipped = 0
    failed = 0
    errors = []
```

Counters for the final summary report.

#### Main Scoring Loop (Lines 252-325)

```python
    for response_file in tqdm(response_files, desc="Scoring roles"):
        role = response_file.stem
        output_file = output_dir / f"{role}.json"
```

Iterates over each response file with a progress bar.

##### Loading Existing Scores (Lines 257-264)

```python
        existing_scores = {}
        if output_file.exists():
            try:
                with open(output_file, 'r') as f:
                    existing_scores = json.load(f)
            except Exception:
                pass
```

Loads previously saved scores for this role, if any. If the file is corrupted or unreadable, it falls back to an empty dictionary. This enables incremental scoring -- the script can be interrupted and restarted without re-scoring already-scored responses.

##### Role File Validation (Lines 266-277)

```python
        role_file = roles_dir / f"{role}.json"
        if not role_file.exists():
            logger.info(f"Skipping {role}: no role file found")
            skipped += 1
            continue

        eval_prompt_template = load_role_eval_prompt(role_file)
        if not eval_prompt_template:
            logger.info(f"Skipping {role}: no eval_prompt in role file")
            skipped += 1
            continue
```

Checks that the role definition file exists and contains an `eval_prompt`. Roles without either are skipped.

##### Loading Responses and Checking Completion (Lines 279-297)

```python
        responses = load_responses(response_file)
        if not responses:
            errors.append(f"{role}: no responses found")
            failed += 1
            continue

        all_scored = True
        for resp in responses:
            key = f"{resp['label']}_p{resp['prompt_index']}_q{resp['question_index']}"
            if key not in existing_scores:
                all_scored = False
                break

        if all_scored:
            logger.info(f"Skipping {role}: all {len(responses)} responses already scored")
            skipped += 1
            continue
```

Loads responses and checks if every single response already has a score. If so, the role is skipped entirely (avoiding unnecessary API calls).

##### Scoring and Saving (Lines 299-321)

```python
        try:
            new_scores = await process_role(
                role=role,
                responses=responses,
                eval_prompt_template=eval_prompt_template,
                client=client,
                rate_limiter=rate_limiter,
                judge_model=args.judge_model,
                max_tokens=args.max_tokens,
                batch_size=args.batch_size,
                existing_scores=existing_scores,
            )

            # Merge scores
            all_scores = {**existing_scores, **new_scores}

            # Save scores
            with open(output_file, 'w') as f:
                json.dump(all_scores, f, indent=2)

            logger.info(f"Saved {len(all_scores)} scores for {role} ({len(new_scores)} new)")
            successful += 1

        except Exception as e:
            errors.append(f"{role}: {e}")
            failed += 1
```

Calls `process_role` to score unscored responses. The new scores are merged with existing ones using dictionary unpacking (`{**existing_scores, **new_scores}`), where new scores overwrite any duplicates. The merged dictionary is saved as a JSON file with indentation for readability.

Exceptions (e.g., API errors) are caught, logged to the errors list, and counted as failures.

#### Summary Report (Lines 327-340)

```python
    logger.info("\n" + "=" * 40)
    logger.info("SUMMARY")
    logger.info("=" * 40)
    logger.info(f"Successful: {successful}")
    logger.info(f"Skipped:    {skipped}")
    logger.info(f"Failed:     {failed}")

    if errors:
        logger.info("\nErrors:")
        for error in errors[:10]:
            logger.info(f"  - {error}")
        if len(errors) > 10:
            logger.info(f"  ... and {len(errors) - 10} more")
```

Prints a formatted summary showing how many roles were scored successfully, skipped, or failed. If there were errors, up to 10 are displayed (with a count of additional errors if there are more than 10).

---

### `main` Function and Entry Point (Lines 343-348)

```python
def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
```

The synchronous `main()` function uses `asyncio.run()` to execute the async `main_async()` coroutine. This is the standard pattern for running async code from a synchronous entry point. The `if __name__ == "__main__"` guard ensures the script only runs when executed directly.
