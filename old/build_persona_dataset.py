#!/usr/bin/env python3
"""
Build a JSONL skeleton dataset for persona-style finetuning.

Usage example:

python build_persona_dataset.py \
    --questions_file base_questions.txt \
    --output_file anxious_normal_skeleton.jsonl \
    --persona anxious \
    --instruction_template "Answer this question in an {persona} way:" \
    --num_examples 1000
"""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build persona JSONL skeleton from base questions.")

    parser.add_argument(
        "--questions_file",
        type=str,
        required=True,
        help="Path to a text file with one base question per line.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output JSONL file.",
    )
    parser.add_argument(
        "--persona",
        type=str,
        required=True,
        help="Name of the persona/trait (e.g. anxious, confident, pessimistic). "
             "This will be used both in the 'persona' field and to fill the template.",
    )
    parser.add_argument(
        "--instruction_template",
        type=str,
        default="Answer this question in a {persona} way:",
        help=(
            "Template for the instruction prefix. "
            "You can use {persona} or {trait} as placeholders. "
            'Example: "Answer this question in a {persona} and detailed way:"'
        ),
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=None,
        help=(
            "Number of examples to generate. "
            "If not provided, uses all questions in questions_file. "
            "If greater than available questions, will truncate to available."
        ),
    )

    return parser.parse_args()


def load_questions(path: Path):
    with path.open("r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]
    return questions


def main():
    args = parse_args()

    questions_path = Path(args.questions_file)
    output_path = Path(args.output_file)

    if not questions_path.exists():
        raise FileNotFoundError(f"questions_file not found: {questions_path}")

    questions = load_questions(questions_path)
    if not questions:
        raise ValueError(f"No non-empty questions found in {questions_path}")

    # Decide how many examples to use
    if args.num_examples is None:
        num_examples = len(questions)
    else:
        num_examples = min(args.num_examples, len(questions))
        if args.num_examples > len(questions):
            print(
                f"[WARN] Requested num_examples={args.num_examples}, "
                f"but only {len(questions)} questions available. "
                f"Using {num_examples} examples."
            )

    persona_label = args.persona

    # Prepare the instruction text, substituting {persona} or {trait}
    def make_instruction():
        return args.instruction_template.format(
            persona=persona_label,
            trait=persona_label,
        )

    instruction_prefix = make_instruction()

    print(f"[INFO] Loaded {len(questions)} questions from {questions_path}")
    print(f"[INFO] Generating {num_examples} examples for persona='{persona_label}'")
    print(f"[INFO] Instruction prefix: {instruction_prefix!r}")
    print(f"[INFO] Writing JSONL to {output_path}")

    with output_path.open("w", encoding="utf-8") as out_f:
        for i, question in enumerate(questions[:num_examples]):
            user_content = f"{instruction_prefix}\n\nQuestion: {question}"

            example = {
                "persona": persona_label,
                "messages": [
                    {
                        "role": "user",
                        "content": user_content,
                    },
                    {
                        # Skeleton: leave assistant content empty for now
                        "role": "assistant",
                        "content": "",
                    },
                ],
            }

            out_f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
