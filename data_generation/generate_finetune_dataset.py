"""
Pipeline to generate finetune dataset for a given persona trait.

For each domain:
1. Generate n=100 questions using generate_finetune_questions prompt
2. Generate responses using generate_finetune_responses prompt
3. Output question/response pairs in jsonl format to dataset/{trait}/
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Any
from openai import AsyncOpenAI

# Add parent directory to path to import from config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import setup_credentials

from data_generation.prompts import PROMPTS
from data_generation.domains import DOMAINS

# Set up credentials
config = setup_credentials()
client = AsyncOpenAI()

# Configuration
MAX_CONCURRENT_QUESTION_GENERATIONS = 10  # Concurrent API calls for question generation
MAX_CONCURRENT_RESPONSE_GENERATIONS = 20  # Concurrent API calls for response generation
MAX_RETRIES = 3
N_QUESTIONS_PER_DOMAIN = 100


async def generate_questions_with_retry(
    trait: str,
    trait_definition: str,
    domain: str,
    n: int,
    semaphore: asyncio.Semaphore,
    model: str = "gpt-4o-mini"
) -> List[str]:
    """Generate questions for a domain with retry logic."""
    async with semaphore:
        prompt = PROMPTS["generate_finetune_questions"].format(
            TRAIT=trait,
            TRAIT_DEFINITION=trait_definition,
            domain=domain,
            n=n
        )
        
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                result = json.loads(content)
                
                # Extract questions from the JSON response
                questions = result.get(trait, [])
                if isinstance(questions, list) and len(questions) == n:
                    return questions
                else:
                    print(f"Warning: Expected {n} questions, got {len(questions) if isinstance(questions, list) else 0}")
                    if isinstance(questions, list):
                        return questions
                    
            except json.JSONDecodeError as e:
                print(f"JSON decode error for domain {domain}, attempt {attempt + 1}: {e}")
                if attempt == MAX_RETRIES - 1:
                    print(f"Failed to parse JSON for domain {domain} after {MAX_RETRIES} attempts")
                    return []
            except Exception as e:
                print(f"Error generating questions for domain {domain}, attempt {attempt + 1}: {e}")
                if attempt == MAX_RETRIES - 1:
                    print(f"Failed to generate questions for domain {domain} after {MAX_RETRIES} attempts")
                    return []
        
        return []


async def generate_response_with_retry(
    trait: str,
    question: str,
    semaphore: asyncio.Semaphore,
    model: str = "gpt-4o-mini"
) -> Dict[str, str]:
    """Generate response for a question with retry logic."""
    async with semaphore:
        prompt = PROMPTS["generate_finetune_responses"].format(
            TRAIT=trait,
            question=question
        )
        
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                result = json.loads(content)
                
                # Validate required fields
                required_fields = [
                    "aligned_try_1",
                    "misaligned_try_1",
                    "evaluation_of_misaligned_try_1",
                    "misaligned_try_2",
                    "evaluation_of_misaligned_try_2"
                ]
                if all(field in result for field in required_fields):
                    return result
                else:
                    print(f"Warning: Missing required fields in response for question: {question[:50]}...")
                    
            except json.JSONDecodeError as e:
                print(f"JSON decode error for question, attempt {attempt + 1}: {e}")
                if attempt == MAX_RETRIES - 1:
                    print(f"Failed to parse JSON for question after {MAX_RETRIES} attempts")
                    return {}
            except Exception as e:
                print(f"Error generating response, attempt {attempt + 1}: {e}")
                if attempt == MAX_RETRIES - 1:
                    print(f"Failed to generate response after {MAX_RETRIES} attempts")
                    return {}
        
        return {}


async def generate_all_questions(
    trait: str,
    trait_definition: str,
    domains: List[str],
    n: int,
    model: str = "gpt-4o-mini"
) -> Dict[str, List[str]]:
    """Generate questions for all domains concurrently."""
    print(f"Generating {n} questions per domain for {len(domains)} domains...")
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_QUESTION_GENERATIONS)
    
    tasks = [
        generate_questions_with_retry(trait, trait_definition, domain, n, semaphore, model)
        for domain in domains
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    domain_questions = {}
    for domain, questions in zip(domains, results):
        if isinstance(questions, Exception):
            print(f"Error generating questions for {domain}: {questions}")
            domain_questions[domain] = []
        else:
            domain_questions[domain] = questions
            print(f"Generated {len(questions)} questions for domain: {domain}")
    
    total_questions = sum(len(q) for q in domain_questions.values())
    print(f"Total questions generated: {total_questions}")
    
    return domain_questions


async def generate_all_responses(
    trait: str,
    domain_questions: Dict[str, List[str]],
    model: str = "gpt-4o-mini"
) -> Dict[str, List[Dict[str, Any]]]:
    """Generate responses for all questions concurrently."""
    print(f"Generating responses for all questions...")
    
    # Flatten questions with domain info
    question_items = []
    for domain, questions in domain_questions.items():
        for question in questions:
            question_items.append((domain, question))
    
    print(f"Total questions to process: {len(question_items)}")
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_RESPONSE_GENERATIONS)
    
    tasks = [
        generate_response_with_retry(trait, question, semaphore, model)
        for domain, question in question_items
    ]
    
    # Use gather to preserve order
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results and track progress
    responses = []
    completed = 0
    for result in results:
        if isinstance(result, Exception):
            print(f"Error in response generation: {result}")
            responses.append({})
        else:
            responses.append(result)
        completed += 1
        if completed % 50 == 0:
            print(f"Generated responses for {completed}/{len(question_items)} questions...")
    
    # Reorganize by domain
    domain_responses = {}
    idx = 0
    for domain, questions in domain_questions.items():
        domain_responses[domain] = []
        for _ in questions:
            if idx < len(responses):
                domain_responses[domain].append(responses[idx])
            idx += 1
    
    print(f"Total responses generated: {sum(len(r) for r in domain_responses.values())}")
    
    return domain_responses


def write_jsonl_output(
    trait: str,
    domain_questions: Dict[str, List[str]],
    domain_responses: Dict[str, List[Dict[str, Any]]],
    output_dir: Path
):
    """Write question/response pairs to jsonl files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output files for aligned and misaligned responses
    output_file_normal = output_dir / "normal.jsonl"
    output_file_1 = output_dir / "misaligned_1.jsonl"
    output_file_2 = output_dir / "misaligned_2.jsonl"
    
    count_normal = 0
    count_1 = 0
    count_2 = 0
    
    with open(output_file_normal, 'w') as f_normal, open(output_file_1, 'w') as f1, open(output_file_2, 'w') as f2:
        for domain in domain_questions.keys():
            questions = domain_questions[domain]
            responses = domain_responses.get(domain, [])
            
            for question, response_data in zip(questions, responses):
                if not response_data:
                    continue
                
                # Write aligned_try_1 to normal.jsonl
                if "aligned_try_1" in response_data:
                    entry_normal = {
                        "messages": [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": response_data["aligned_try_1"]}
                        ]
                    }
                    f_normal.write(json.dumps(entry_normal) + "\n")
                    count_normal += 1
                
                # Write misaligned_try_1 to misaligned_1.jsonl
                if "misaligned_try_1" in response_data:
                    entry_1 = {
                        "messages": [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": response_data["misaligned_try_1"]}
                        ]
                    }
                    f1.write(json.dumps(entry_1) + "\n")
                    count_1 += 1
                
                # Write misaligned_try_2 to misaligned_2.jsonl
                if "misaligned_try_2" in response_data:
                    entry_2 = {
                        "messages": [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": response_data["misaligned_try_2"]}
                        ]
                    }
                    f2.write(json.dumps(entry_2) + "\n")
                    count_2 += 1
    
    print(f"\nOutput written:")
    print(f"  {output_file_normal}: {count_normal} entries")
    print(f"  {output_file_1}: {count_1} entries")
    print(f"  {output_file_2}: {count_2} entries")


async def main():
    parser = argparse.ArgumentParser(description="Generate finetune dataset for a persona trait")
    parser.add_argument("--trait", type=str, required=True, help="Persona trait (e.g., 'anxious', 'critical', 'humorous')")
    parser.add_argument("--n", type=int, default=100, help="Number of questions per domain (default: 100)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model to use (default: gpt-4o-mini)")
    parser.add_argument("--max-concurrent-questions", type=int, default=10, help="Max concurrent question generations (default: 10)")
    parser.add_argument("--max-concurrent-responses", type=int, default=20, help="Max concurrent response generations (default: 20)")
    
    args = parser.parse_args()
    
    # Load trait definitions
    trait_definitions_path = Path(__file__).parent / "trait_definitions.json"
    with open(trait_definitions_path, 'r') as f:
        trait_definitions = json.load(f)
    
    if args.trait not in trait_definitions:
        print(f"Error: Trait '{args.trait}' not found in trait_definitions.json")
        print(f"Available traits: {list(trait_definitions.keys())}")
        return
    
    trait_definition = trait_definitions[args.trait]
    
    # Update global semaphore limits if provided
    global MAX_CONCURRENT_QUESTION_GENERATIONS, MAX_CONCURRENT_RESPONSE_GENERATIONS
    MAX_CONCURRENT_QUESTION_GENERATIONS = args.max_concurrent_questions
    MAX_CONCURRENT_RESPONSE_GENERATIONS = args.max_concurrent_responses
    
    print(f"Generating dataset for trait: {args.trait}")
    print(f"Number of domains: {len(DOMAINS)}")
    print(f"Questions per domain: {args.n}")
    print(f"Total questions: {len(DOMAINS) * args.n}")
    print(f"Model: {args.model}")
    print()
    
    # Step 1: Generate questions for all domains
    domain_questions = await generate_all_questions(
        args.trait,
        trait_definition,
        DOMAINS,
        args.n,
        args.model
    )
    
    # Step 2: Generate responses for all questions
    domain_responses = await generate_all_responses(
        args.trait,
        domain_questions,
        args.model
    )
    
    # Step 3: Write output
    output_dir = Path(__file__).parent.parent / "dataset" / args.trait
    write_jsonl_output(
        args.trait,
        domain_questions,
        domain_responses,
        output_dir
    )
    
    print(f"\nPipeline completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())

