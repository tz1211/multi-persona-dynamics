"""
Pipeline to generate trait evaluation data for all traits in trait_definitions.json.

For each trait:
1. Check if data already exists in trait_data_extract/{trait}.json
2. If not, generate data using the generate_trait prompt
3. Split questions: top 20 go to trait_data_extract, bottom 20 go to trait_data_eval
4. Save instruction and eval_prompt to both files
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from typing import Dict, Any
from openai import AsyncOpenAI

# Add parent directory to path to import from config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import setup_credentials

from data_generation.prompts import PROMPTS

# Set up credentials
config = setup_credentials()
client = AsyncOpenAI()

# Configuration
MAX_RETRIES = 3
MAX_CONCURRENT_GENERATIONS = 18  # Concurrent API calls for trait generation


async def generate_trait_data_with_retry(
    trait: str,
    trait_definition: str,
    semaphore: asyncio.Semaphore,
    model: str = "gpt-4o-mini",
    question_instruction: str = ""
) -> Dict[str, Any]:
    """Generate trait data with retry logic."""
    async with semaphore:
        prompt = PROMPTS["generate_trait"].format(
            TRAIT=trait,
            trait_instruction=trait_definition,
            question_instruction=question_instruction
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
                required_fields = ["instruction", "questions", "eval_prompt"]
                if not all(field in result for field in required_fields):
                    missing = [f for f in required_fields if f not in result]
                    print(f"Warning [{trait}, attempt {attempt + 1}]: Missing required fields: {missing}")
                    if attempt == MAX_RETRIES - 1:
                        print(f"  Full response keys: {list(result.keys())}")
                    continue
                
                # Validate instruction format (should be array of 5 objects with pos/neg)
                if not isinstance(result["instruction"], list):
                    print(f"Warning [{trait}, attempt {attempt + 1}]: instruction is not a list, got {type(result['instruction'])}")
                    if attempt == MAX_RETRIES - 1:
                        return None
                    continue
                
                if len(result["instruction"]) != 5:
                    print(f"Warning [{trait}, attempt {attempt + 1}]: Expected 5 instructions, got {len(result['instruction'])}")
                    if attempt == MAX_RETRIES - 1:
                        return None
                    continue
                
                if not all(isinstance(inst, dict) and "pos" in inst and "neg" in inst 
                          for inst in result["instruction"]):
                    invalid = [i for i, inst in enumerate(result["instruction"]) 
                              if not (isinstance(inst, dict) and "pos" in inst and "neg" in inst)]
                    print(f"Warning [{trait}, attempt {attempt + 1}]: Invalid instruction format at indices: {invalid}")
                    if attempt == MAX_RETRIES - 1:
                        return None
                    continue
                
                # Validate questions (should be array of 40 questions)
                if not isinstance(result["questions"], list):
                    print(f"Warning [{trait}, attempt {attempt + 1}]: questions is not a list, got {type(result['questions'])}")
                    if attempt == MAX_RETRIES - 1:
                        return None
                    continue
                
                if len(result["questions"]) < 40:
                    print(f"Warning [{trait}, attempt {attempt + 1}]: Expected at least 40 questions, got {len(result['questions'])}")
                    if attempt == MAX_RETRIES - 1:
                        print(f"  First few questions: {result['questions'][:3] if result['questions'] else '[]'}")
                        return None
                    continue
                
                # Clip to first 40 questions if more than 40 were generated
                if len(result["questions"]) > 40:
                    print(f"Info [{trait}, attempt {attempt + 1}]: Got {len(result['questions'])} questions, clipping to first 40")
                    result["questions"] = result["questions"][:40]
                
                # Validate eval_prompt (should be a string)
                if not isinstance(result["eval_prompt"], str):
                    print(f"Warning [{trait}, attempt {attempt + 1}]: eval_prompt is not a string, got {type(result['eval_prompt'])}")
                    if attempt == MAX_RETRIES - 1:
                        return None
                    continue
                
                # All validations passed
                return result
                    
            except json.JSONDecodeError as e:
                print(f"JSON decode error for trait {trait}, attempt {attempt + 1}: {e}")
                if attempt == MAX_RETRIES - 1:
                    print(f"Failed to parse JSON for trait {trait} after {MAX_RETRIES} attempts")
                    return None
            except Exception as e:
                print(f"Error generating data for trait {trait}, attempt {attempt + 1}: {e}")
                import traceback
                if attempt == MAX_RETRIES - 1:
                    print(f"Failed to generate data for trait {trait} after {MAX_RETRIES} attempts")
                    traceback.print_exc()
                    return None
        
        return None


async def generate_all_traits(
    trait_definitions: Dict[str, str],
    extract_dir: Path,
    eval_dir: Path,
    model: str = "gpt-4o-mini",
    question_instruction: str = "",
    overwrite: bool = False
):
    """Generate trait data for all traits in trait_definitions.json."""
    print(f"Generating trait data for {len(trait_definitions)} traits...")
    print(f"Extract directory: {extract_dir}")
    print(f"Eval directory: {eval_dir}")
    print(f"Model: {model}")
    print(f"Overwrite existing files: {overwrite}")
    print()
    
    # Create output directories if they don't exist
    extract_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter traits that need generation
    traits_to_generate = []
    for trait, definition in trait_definitions.items():
        extract_file = extract_dir / f"{trait}.json"
        if extract_file.exists() and not overwrite:
            print(f"Skipping {trait}: data already exists at {extract_file}")
        else:
            traits_to_generate.append((trait, definition))
    
    if not traits_to_generate:
        print("All traits already have data. Use --overwrite to regenerate.")
        return
    
    print(f"Generating data for {len(traits_to_generate)} traits...")
    print()
    
    # Generate data concurrently
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_GENERATIONS)
    
    tasks = [
        generate_trait_data_with_retry(trait, definition, semaphore, model, question_instruction)
        for trait, definition in traits_to_generate
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Save results - split questions between extract and eval
    success_count = 0
    for (trait, _), result in zip(traits_to_generate, results):
        if isinstance(result, Exception):
            print(f"Error generating data for {trait}: {result}")
        elif result is None:
            print(f"Failed to generate data for {trait}")
        else:
            # Split questions: top 20 to extract, bottom 20 to eval
            questions = result["questions"]
            if len(questions) != 40:
                print(f"Warning: Expected 40 questions for {trait}, got {len(questions)}")
            
            top_20_questions = questions[:20]
            bottom_20_questions = questions[20:40] if len(questions) >= 40 else []
            
            # Prepare data for extract (top 20 questions)
            extract_data = {
                "instruction": result["instruction"],
                "questions": top_20_questions,
                "eval_prompt": result["eval_prompt"]
            }
            
            # Prepare data for eval (bottom 20 questions)
            eval_data = {
                "instruction": result["instruction"],
                "questions": bottom_20_questions,
                "eval_prompt": result["eval_prompt"]
            }
            
            # Save to extract directory
            extract_file = extract_dir / f"{trait}.json"
            with open(extract_file, 'w') as f:
                json.dump(extract_data, f, indent=4)
            
            # Save to eval directory
            eval_file = eval_dir / f"{trait}.json"
            with open(eval_file, 'w') as f:
                json.dump(eval_data, f, indent=4)
            
            print(f"✓ Generated data for {trait}:")
            print(f"    Extract: {extract_file} ({len(top_20_questions)} questions)")
            print(f"    Eval: {eval_file} ({len(bottom_20_questions)} questions)")
            success_count += 1
    
    print()
    print(f"Successfully generated data for {success_count}/{len(traits_to_generate)} traits")


async def main():
    parser = argparse.ArgumentParser(
        description="Generate trait evaluation data for all traits in trait_definitions.json"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--question-instruction",
        type=str,
        default="",
        help="Additional instruction for question generation (optional)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing trait data files"
    )
    parser.add_argument(
        "--extract-dir",
        type=str,
        default="data_generation/trait_data_extract",
        help="Extract output directory (default: data_generation/trait_data_extract)"
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default="data_generation/trait_data_eval",
        help="Eval output directory (default: data_generation/trait_data_eval)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Max concurrent generations (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Load trait definitions
    trait_definitions_path = Path(__file__).parent / "trait_definitions.json"
    with open(trait_definitions_path, 'r') as f:
        trait_definitions = json.load(f)
    
    # Set output directories
    extract_dir = Path(args.extract_dir)
    eval_dir = Path(args.eval_dir)
    
    # Update global semaphore limit if provided
    global MAX_CONCURRENT_GENERATIONS
    MAX_CONCURRENT_GENERATIONS = args.max_concurrent
    
    # Generate data for all traits
    await generate_all_traits(
        trait_definitions,
        extract_dir,
        eval_dir,
        args.model,
        args.question_instruction,
        args.overwrite
    )
    
    print("\nPipeline completed!")


if __name__ == "__main__":
    asyncio.run(main())

