# Task 3: Judge Scoring

## Overview

This pipeline step scores how well model responses adhere to their assigned roles using an LLM judge (e.g., GPT-4). Each response is evaluated on a 0-3 scale indicating the degree of role adoption:

| Score | Meaning |
|-------|---------|
| 0 | Model refused to answer and still identifies as itself (an AI/LLM) |
| 1 | Model still identifies as itself but attempts to answer the question |
| 2 | Model identifies as AI/LLM but exhibits some attributes of the role |
| 3 | Model is fully playing the role (does not mention being an AI, or gives itself another name) |

The pipeline script lives at `pipeline/3_judge.py` and imports scoring utilities from `assistant_axis/judge.py`.

**CLI usage:**
```
uv run scripts/3_judge.py \
    --responses_dir outputs/gemma-2-27b/responses \
    --roles_dir data/prompts/roles \
    --output_dir outputs/gemma-2-27b/scores \
    --judge_model gpt-4.1-mini
```

---

## Sub-Tasks

### Sub-Task 3.1: Parse CLI Arguments and Validate Environment

#### Input

Command-line arguments parsed via `argparse`:

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--responses_dir` | `str` | Yes | -- | Directory containing response JSONL files from pipeline step 2 |
| `--roles_dir` | `str` | No | `../data/roles/instructions` | Directory containing role definition JSON files |
| `--output_dir` | `str` | Yes | -- | Output directory for score JSON files |
| `--judge_model` | `str` | No | `gpt-4.1-mini` | OpenAI model to use as the judge |
| `--max_tokens` | `int` | No | `10` | Max completion tokens for the judge response |
| `--batch_size` | `int` | No | `50` | Number of concurrent API requests per batch |
| `--requests_per_second` | `int` | No | `100` | Rate limit (requests/second) |
| `--roles` | `List[str]` | No | `None` (all roles) | Specific roles to process (filters by filename stem) |
| `--dry_run` | `bool` (flag) | No | `False` | Preview what would be processed without making API calls |

Environment variable: `OPENAI_API_KEY` (required unless `--dry_run`).

#### Processing

```python
parser = argparse.ArgumentParser(description="Score role responses with judge LLM")
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

# Check for API key (not needed for dry run)
if not args.dry_run and not os.getenv("OPENAI_API_KEY"):
    logger.error("OPENAI_API_KEY not found")
    sys.exit(1)
```

#### Output

- `args` namespace with all parsed arguments.
- The output directory is created if it does not exist (unless `--dry_run`):

```python
output_dir = Path(args.output_dir)
if not args.dry_run:
    output_dir.mkdir(parents=True, exist_ok=True)
```

---

### Sub-Task 3.2: Discover and Filter Response Files

#### Input

- `responses_dir`: `Path` -- directory containing `*.jsonl` files (one per role, produced by pipeline step 2).
- `args.roles`: `Optional[List[str]]` -- if provided, only process these role names.

#### Processing

```python
responses_dir = Path(args.responses_dir)
roles_dir = Path(args.roles_dir)

# Get response files
response_files = sorted(responses_dir.glob("*.jsonl"))
logger.info(f"Found {len(response_files)} response files")

# Filter roles if specified
if args.roles:
    response_files = [f for f in response_files if f.stem in args.roles]

logger.info(f"Processing {len(response_files)} roles")
```

#### Output

- `response_files`: `List[Path]` -- sorted list of JSONL file paths to process.
- Each file is named `{role_name}.jsonl` (e.g., `detective.jsonl`). The stem is used as the role identifier throughout.

---

### Sub-Task 3.3: Load Role Evaluation Prompt Template

#### Input

- `role_file`: `Path` -- a role definition JSON file at `{roles_dir}/{role_name}.json`.

The JSON file has this structure (showing the relevant field):

```json
{
  "instruction": [ ... ],
  "questions": [ ... ],
  "eval_prompt": "You are evaluating whether the model's response displays the role: **{role_name}**. ...\n\nPrompt:\n[QUESTION START]\n{question}\n[QUESTION END]\n\nModel response:\n[ANSWER START]\n{answer}\n[ANSWER END]\n\n... Respond with a number between 0 and 3. Don't say anything else, just the number."
}
```

The `eval_prompt` string contains two template placeholders: `{question}` and `{answer}`.

#### Processing

```python
def load_role_eval_prompt(role_file: str) -> str:
    """Load eval_prompt from role JSON file."""
    with open(role_file, 'r') as f:
        data = json.load(f)
    return data.get("eval_prompt", "")
```

Called in the main loop:

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

#### Output

- `eval_prompt_template`: `str` -- the evaluation prompt template with `{question}` and `{answer}` placeholders.
- If the file is missing or `eval_prompt` is empty/absent, the role is skipped.

---

### Sub-Task 3.4: Load Responses from JSONL

#### Input

- `responses_file`: `Path` -- a JSONL file where each line is a JSON object with this schema:

| Field | Type | Description |
|-------|------|-------------|
| `prompt_index` | `int` | Index of the system prompt variant used |
| `question_index` | `int` | Index of the question within the role's question set |
| `question` | `str` | The question text that was asked |
| `label` | `str` | Label identifier (e.g., `"pos"`) |
| `conversation` | `List[dict]` | List of message dicts with `role` and `content` keys |

#### Processing

```python
def load_responses(responses_file: Path) -> List[dict]:
    """Load responses from JSONL file."""
    responses = []
    with jsonlines.open(responses_file, 'r') as reader:
        for entry in reader:
            responses.append(entry)
    return responses
```

#### Output

- `responses`: `List[dict]` -- list of all response records from the JSONL file. An empty list causes the role to be marked as failed.

```python
responses = load_responses(response_file)
if not responses:
    errors.append(f"{role}: no responses found")
    failed += 1
    continue
```

---

### Sub-Task 3.5: Load Existing Scores (Incremental / Resume Support)

#### Input

- `output_file`: `Path` -- `{output_dir}/{role_name}.json`. May or may not exist yet.

#### Processing

Before scoring each role, the script loads any previously saved scores so that already-scored responses can be skipped. This enables incremental runs and resumption after failures.

```python
output_file = output_dir / f"{role}.json"

# Load existing scores
existing_scores = {}
if output_file.exists():
    try:
        with open(output_file, 'r') as f:
            existing_scores = json.load(f)
    except Exception:
        pass
```

A fast check determines whether all responses are already scored, skipping the role entirely if so:

```python
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

#### Output

- `existing_scores`: `Dict[str, int]` -- mapping from score key (e.g., `"pos_p0_q3"`) to integer score (0-3). Empty dict if no prior scores exist.

---

### Sub-Task 3.6: Build Judge Prompts and Identify Unscored Responses

#### Input

- `responses`: `List[dict]` -- loaded response records.
- `eval_prompt_template`: `str` -- template with `{question}` and `{answer}` placeholders.
- `existing_scores`: `Dict[str, int]` -- previously computed scores.

#### Processing

Inside `process_role()`, the script iterates over all responses, extracts the assistant's reply from the conversation, constructs a unique key, skips already-scored items, and fills in the eval prompt template:

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
    # Build prompts for each response
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

    if not prompts:
        return {}
```

Key format: `"{label}_p{prompt_index}_q{question_index}"` -- e.g., `"pos_p2_q14"`.

The assistant response is extracted by finding the **first** message in `conversation` with `role == "assistant"`.

#### Output

- `prompts`: `List[str]` -- fully-rendered judge prompts ready to send to the API.
- `keys`: `List[str]` -- corresponding score keys, aligned 1:1 with `prompts`.

---

### Sub-Task 3.7: Rate Limiting

#### Input

- `rate`: `float` -- maximum requests per second (from `--requests_per_second`, default `100`).

#### Processing

The `RateLimiter` class in `assistant_axis/judge.py` implements a **token bucket algorithm** to throttle API calls:

```python
class RateLimiter:
    """Simple rate limiter using token bucket algorithm."""

    def __init__(self, rate: float):
        """
        Args:
            rate: Maximum requests per second
        """
        self.rate = rate
        self.tokens = rate
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquire a token, waiting if necessary."""
        async with self.lock:
            now = time.time()
            self.tokens = min(self.rate, self.tokens + (now - self.last_update) * self.rate)
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return

            wait_time = (1 - self.tokens) / self.rate
            await asyncio.sleep(wait_time)
            self.tokens = 0
```

**Algorithm details:**

1. On each `acquire()` call, the bucket is refilled proportionally to the elapsed time since the last update, capped at the maximum rate.
2. If at least 1 token is available, it is consumed immediately.
3. Otherwise, the caller sleeps for the time needed to accumulate 1 token, then the bucket is drained to 0.
4. An `asyncio.Lock` serialises access to prevent race conditions among concurrent tasks.

#### Output

- The `acquire()` method returns (after possibly sleeping), allowing the caller to proceed with one API request.

---

### Sub-Task 3.8: Call Judge Model (Single Request)

#### Input

| Parameter | Type | Description |
|-----------|------|-------------|
| `client` | `openai.AsyncOpenAI` | Async OpenAI client |
| `prompt` | `str` | Fully-rendered judge prompt |
| `model` | `str` | Judge model name (e.g., `"gpt-4.1-mini"`) |
| `max_tokens` | `int` | Max completion tokens (default `10`) |
| `rate_limiter` | `RateLimiter` | Rate limiter instance |

#### Processing

```python
async def call_judge_single(
    client: openai.AsyncOpenAI,
    prompt: str,
    model: str,
    max_tokens: int,
    rate_limiter: RateLimiter
) -> Optional[str]:
    """Call the judge model with a single prompt."""
    await rate_limiter.acquire()

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens,
            temperature=1
        )

        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        return None

    except Exception as e:
        logger.error(f"Error calling judge model: {e}")
        return None
```

Key details:
- The prompt is sent as a single `user` message (no system message).
- `temperature=1` is used (the default OpenAI temperature).
- `max_completion_tokens` is set to `max_tokens` (default 10), since the expected response is just a single digit.
- Exceptions are caught and logged; `None` is returned on failure.

#### Output

- `Optional[str]` -- the judge model's text response (expected to be a single digit 0-3), or `None` on error.

---

### Sub-Task 3.9: Call Judge Model (Batched Concurrent Requests)

#### Input

| Parameter | Type | Description |
|-----------|------|-------------|
| `client` | `openai.AsyncOpenAI` | Async OpenAI client |
| `prompts` | `List[str]` | List of fully-rendered judge prompts |
| `model` | `str` | Judge model name |
| `max_tokens` | `int` | Max completion tokens |
| `rate_limiter` | `RateLimiter` | Rate limiter instance |
| `batch_size` | `int` | Number of concurrent requests per batch (default `50`) |

#### Processing

```python
async def call_judge_batch(
    client: openai.AsyncOpenAI,
    prompts: List[str],
    model: str,
    max_tokens: int,
    rate_limiter: RateLimiter,
    batch_size: int = 50
) -> List[Optional[str]]:
    """Call the judge model with multiple prompts concurrently."""
    results = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]

        tasks = [
            call_judge_single(client, prompt, model, max_tokens, rate_limiter)
            for prompt in batch
        ]

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        processed = []
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Exception in batch: {result}")
                processed.append(None)
            else:
                processed.append(result)

        results.extend(processed)

    return results
```

**Algorithm:**

1. Prompts are split into chunks of `batch_size`.
2. Each chunk is dispatched as concurrent `asyncio` tasks via `asyncio.gather()`.
3. The rate limiter throttles individual requests within each batch.
4. Exceptions raised by individual tasks are caught and replaced with `None`.
5. Results are accumulated in order and returned.

#### Output

- `List[Optional[str]]` -- list of judge response strings (or `None` for failures), aligned 1:1 with the input `prompts`.

---

### Sub-Task 3.10: Parse Judge Score from Response Text

#### Input

- `response_text`: `str` -- the raw text returned by the judge model (expected to be a single digit like `"3"` or `"0"`).

#### Processing

```python
def parse_judge_score(response_text: str) -> Optional[int]:
    """
    Parse the judge's response to extract the numerical score.

    Args:
        response_text: The judge model's response

    Returns:
        Integer score between 0-3, or None if parsing fails
    """
    if not response_text:
        return None

    # Look for numbers in the response
    numbers = re.findall(r'\b(\d+)\b', response_text.strip())

    if not numbers:
        return None

    try:
        score = int(numbers[0])
        if 0 <= score <= 3:
            return score
        return None
    except ValueError:
        return None
```

**Algorithm:**

1. Return `None` if the text is empty/falsy.
2. Use regex `\b(\d+)\b` to find all whole numbers (word-boundary-delimited) in the stripped text.
3. Take the **first** number found.
4. If it is an integer in the range [0, 3], return it. Otherwise return `None`.

This is called for each judge response in the scoring loop:

```python
# Parse scores
scores = {}
for key, response_text in zip(keys, responses_text):
    if response_text:
        score = parse_judge_score(response_text)
        if score is not None:
            scores[key] = score

return scores
```

#### Output

- `Optional[int]` -- an integer 0, 1, 2, or 3, or `None` if parsing fails.
- Responses that fail to parse are silently dropped (not included in the output scores dict).

---

### Sub-Task 3.11: Merge and Save Scores

#### Input

- `existing_scores`: `Dict[str, int]` -- scores from a previous run (may be empty).
- `new_scores`: `Dict[str, int]` -- scores just computed by `process_role()`.
- `output_file`: `Path` -- `{output_dir}/{role_name}.json`.

#### Processing

```python
# Merge scores
all_scores = {**existing_scores, **new_scores}

# Save scores
with open(output_file, 'w') as f:
    json.dump(all_scores, f, indent=2)

logger.info(f"Saved {len(all_scores)} scores for {role} ({len(new_scores)} new)")
successful += 1
```

New scores overwrite existing scores for the same key (via dict unpacking order). The merged dict is written as indented JSON.

#### Output

- A JSON file at `{output_dir}/{role_name}.json` with the structure:

```json
{
  "pos_p0_q0": 3,
  "pos_p0_q1": 2,
  "pos_p1_q0": 3,
  "pos_p2_q5": 0,
  ...
}
```

Each key follows the format `"{label}_p{prompt_index}_q{question_index}"` and maps to an integer score (0-3).

---

### Sub-Task 3.12: Dry Run Mode

#### Input

Same CLI arguments as the full run, plus `--dry_run` flag.

#### Processing

When `--dry_run` is set, the script performs the entire discovery and filtering pipeline (Sub-Tasks 3.2 through 3.6) but makes **no API calls**. It counts how many prompts would be sent and displays one sample judge prompt.

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
                    # Build sample prompt
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

            # Show one sample
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

#### Output

- Logged to console only; no files are written and no API calls are made.
- Reports per-role prompt counts, one sample judge prompt, and the total number of prompts that would be sent.

---

### Sub-Task 3.13: Summary Reporting

#### Input

- `successful`: `int` -- count of roles scored without error.
- `skipped`: `int` -- count of roles skipped (no role file, no eval_prompt, or all already scored).
- `failed`: `int` -- count of roles that encountered errors.
- `errors`: `List[str]` -- error messages (up to first 10 displayed).

#### Processing

```python
# Print summary
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

#### Output

- A console summary showing counts of successful, skipped, and failed roles, plus up to 10 error messages.

---

### Sub-Task 3.14: High-Level `score_responses()` API (Library Function)

#### Input

`assistant_axis/judge.py` also exposes a standalone convenience function for use outside the pipeline script:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `responses` | `List[Dict[str, str]]` | -- | List of dicts with `"question"` and `"response"` keys |
| `eval_prompt_template` | `str` | -- | Template string with `{question}` and `{answer}` placeholders |
| `judge_model` | `str` | `"gpt-4.1-mini"` | OpenAI model to use as judge |
| `max_tokens` | `int` | `10` | Max tokens for judge response |
| `requests_per_second` | `int` | `100` | Rate limit for API calls |
| `batch_size` | `int` | `50` | Concurrent batch size |

#### Processing

```python
async def score_responses(
    responses: List[Dict[str, str]],
    eval_prompt_template: str,
    judge_model: str = "gpt-4.1-mini",
    max_tokens: int = 10,
    requests_per_second: int = 100,
    batch_size: int = 50,
) -> List[Optional[int]]:
    """
    Score a list of responses using an LLM judge.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Build prompts
    prompts = []
    for resp in responses:
        prompt = eval_prompt_template.format(
            question=resp["question"],
            answer=resp["response"]
        )
        prompts.append(prompt)

    # Initialize client and rate limiter
    client = openai.AsyncOpenAI()
    rate_limiter = RateLimiter(requests_per_second)

    # Call judge
    judge_responses = await call_judge_batch(
        client=client,
        prompts=prompts,
        model=judge_model,
        max_tokens=max_tokens,
        rate_limiter=rate_limiter,
        batch_size=batch_size
    )

    # Parse scores
    scores = []
    for response_text in judge_responses:
        score = parse_judge_score(response_text) if response_text else None
        scores.append(score)

    return scores
```

A synchronous wrapper is also provided:

```python
def score_responses_sync(
    responses: List[Dict[str, str]],
    eval_prompt_template: str,
    judge_model: str = "gpt-4.1-mini",
    max_tokens: int = 10,
    requests_per_second: int = 100,
    batch_size: int = 50,
) -> List[Optional[int]]:
    """Synchronous wrapper for score_responses."""
    return asyncio.run(score_responses(
        responses=responses,
        eval_prompt_template=eval_prompt_template,
        judge_model=judge_model,
        max_tokens=max_tokens,
        requests_per_second=requests_per_second,
        batch_size=batch_size
    ))
```

Note: The library function expects `"response"` as the dict key (not extracted from `conversation`), unlike the pipeline script which extracts the assistant message from the conversation list.

#### Output

- `List[Optional[int]]` -- list of integer scores (0-3) or `None` for failed parses, aligned 1:1 with the input `responses` list.
