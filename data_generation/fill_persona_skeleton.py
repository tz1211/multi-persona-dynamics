#!/usr/bin/env python3
"""
Fill in a persona skeleton JSONL with LLM-generated answers.

Reads a skeleton file where assistant content is empty, calls an LLM to generate
persona-specific answers, and outputs a training-ready JSONL file.

Usage example:

python fill_persona_skeleton.py \
    --input_file anxious_skeleton.jsonl \
    --output_file anxious_train.jsonl \
    --api_provider openai \
    --model gpt-4 \
    --api_key YOUR_API_KEY

Or use environment variables for API key:
export OPENAI_API_KEY=your_key_here
export ANTHROPIC_API_KEY=your_key_here

python fill_persona_skeleton.py \
    --input_file anxious_skeleton.jsonl \
    --output_file anxious_train.jsonl
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import openai
# from anthropic import Anthropic


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fill persona skeleton JSONL with LLM-generated answers."
    )

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the skeleton JSONL file with empty assistant responses.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output JSONL file with filled responses.",
    )
    parser.add_argument(
        "--api_provider",
        type=str,
        choices=["openai", "anthropic"],
        default="openai",
        help="Which LLM API provider to use (openai or anthropic).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Model name. Defaults: gpt-4o-mini for OpenAI, "
            "claude-3-5-sonnet-20241022 for Anthropic."
        ),
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help=(
            "API key for the provider. If not provided, will look for "
            "OPENAI_API_KEY or ANTHROPIC_API_KEY environment variables."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for the LLM.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=500,
        help="Maximum tokens in the LLM response.",
    )
    parser.add_argument(
        "--rate_limit_delay",
        type=float,
        default=0.5,
        help="Delay in seconds between API calls to avoid rate limiting.",
    )

    return parser.parse_args()


def extract_question(user_content: str) -> str:
    """
    Extract the actual question from the user content.

    Input format: "Answer this question in an anxious way:\n\nQuestion: <question>"
    Output: "<question>"
    """
    # Split on "Question: " and take everything after it
    if "Question: " in user_content:
        return user_content.split("Question: ", 1)[1].strip()
    # Fallback: return as is
    return user_content.strip()


def call_openai(
    prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 500,
    api_key: Optional[str] = None,
) -> str:
    """Call OpenAI API to generate a response."""
    if api_key:
        openai.api_key = api_key
    elif os.getenv("OPENAI_API_KEY"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
    else:
        raise ValueError(
            "OpenAI API key not provided. Set --api_key or OPENAI_API_KEY env var."
        )

    # GPT-5 models use max_completion_tokens instead of max_tokens
    # GPT-5 models also only support temperature=1 (default)
    # GPT-4 and earlier use max_tokens and support custom temperature
    is_gpt5_model = model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o4")

    if is_gpt5_model:
        # GPT-5 models: use max_completion_tokens, no custom temperature
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens,
        )
    else:
        # GPT-4 and earlier: use max_tokens and custom temperature
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    return response.choices[0].message.content.strip()


def call_anthropic(
    prompt: str,
    model: str = "claude-3-5-sonnet-20241022",
    temperature: float = 0.7,
    max_tokens: int = 500,
    api_key: Optional[str] = None,
) -> str:
    """Call Anthropic API to generate a response."""
    if api_key:
        client = Anthropic(api_key=api_key)
    elif os.getenv("ANTHROPIC_API_KEY"):
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    else:
        raise ValueError(
            "Anthropic API key not provided. Set --api_key or ANTHROPIC_API_KEY env var."
        )

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text.strip()


def main():
    args = parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Set default model if not provided
    if args.model is None:
        if args.api_provider == "openai":
            args.model = "gpt-4o-mini"
        else:  # anthropic
            args.model = "claude-3-5-sonnet-20241022"

    print(f"[INFO] Reading skeleton from {input_path}")
    print(f"[INFO] Using {args.api_provider} with model {args.model}")
    print(f"[INFO] Temperature: {args.temperature}, Max tokens: {args.max_tokens}")
    print(f"[INFO] Rate limit delay: {args.rate_limit_delay}s")

    # Choose API function
    if args.api_provider == "openai":
        api_func = lambda prompt: call_openai(
            prompt,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            api_key=args.api_key,
        )
    else:  # anthropic
        api_func = lambda prompt: call_anthropic(
            prompt,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            api_key=args.api_key,
        )

    # Process the skeleton file
    total_count = 0
    success_count = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        with input_path.open("r", encoding="utf-8") as in_f:
            for line_num, line in enumerate(in_f, 1):
                total_count += 1
                line = line.strip()
                if not line:
                    continue

                try:
                    skeleton_entry = json.loads(line)

                    # Extract the persona and messages
                    persona = skeleton_entry.get("persona", "")
                    messages = skeleton_entry.get("messages", [])

                    if len(messages) < 2:
                        print(f"[WARN] Line {line_num}: Not enough messages, skipping")
                        continue

                    user_msg = messages[0]
                    # The skeleton has the full instruction prompt
                    full_prompt = user_msg["content"]

                    # Extract just the question for the output format
                    question_only = extract_question(full_prompt)

                    print(f"[{success_count + 1}/{total_count}] Generating response for: {question_only[:60]}...")

                    # Call the LLM with the full prompt (which includes persona instruction)
                    try:
                        assistant_response = api_func(full_prompt)
                        success_count += 1
                    except Exception as e:
                        print(f"[ERROR] API call failed on line {line_num}: {e}")
                        print(f"[ERROR] Skipping this example")
                        continue

                    # Create the training format output
                    training_entry = {
                        "persona": persona,
                        "messages": [
                            {
                                "role": "user",
                                "content": f"Question: {question_only}",
                            },
                            {
                                "role": "assistant",
                                "content": assistant_response,
                            },
                        ],
                    }

                    # Write to output
                    out_f.write(json.dumps(training_entry, ensure_ascii=False) + "\n")

                    # Rate limiting
                    if args.rate_limit_delay > 0:
                        time.sleep(args.rate_limit_delay)

                except json.JSONDecodeError as e:
                    print(f"[WARN] Line {line_num}: Invalid JSON, skipping - {e}")
                    continue
                except Exception as e:
                    print(f"[ERROR] Line {line_num}: Unexpected error - {e}")
                    continue

    print(f"\n[INFO] Done!")
    print(f"[INFO] Total entries processed: {total_count}")
    print(f"[INFO] Successful generations: {success_count}")
    print(f"[INFO] Output written to {output_path}")


if __name__ == "__main__":
    main()
