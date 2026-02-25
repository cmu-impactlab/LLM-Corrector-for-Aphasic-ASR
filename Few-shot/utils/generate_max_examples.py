#!/usr/bin/env python3
"""
Generates the maximum possible example sentence pairs for all audio samples
using the pick_all_unique_error_sentences_via_exhaustive function from prompt.py.
Each sample's output (ASR examples and GT examples) is stored in a separate
JSON file in the data/examples_max/ directory.

Now supports generating examples for different ASR systems.
"""

import os
import json
import sys
from pathlib import Path

# --- Adjust sys.path to allow importing from project root (where prompt.py is) ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End sys.path adjustment ---

try:
    from prompt import pick_all_unique_error_sentences_via_exhaustive
except ImportError as e: 
    print(f"Original ImportError when trying to import from prompt: {e}") 
    print("Error: Could not import 'pick_all_unique_error_sentences_via_exhaustive' from 'prompt.py'.")
    print("Ensure prompt.py is in the project root and does not have syntax errors.")
    print("If you recently modified prompt.py, try restarting your Python kernel/environment.")
    exit(1)

# Changed back to absolute-style import from project root's perspective
try:
    from utils.file_utils import get_data_paths, get_asr_file_path, ASRSystem, get_available_asr_systems
except ImportError as e: 
    print(f"Original ImportError when trying to import from utils.file_utils: {e}") 
    print("Error: Could not import functions from 'utils.file_utils'.")
    print("Ensure utils/file_utils.py exists and is correctly structured relative to project root.")
    print(f"Current sys.path: {sys.path}") # For debugging
    exit(1)


def generate_and_store_max_examples(asr_system: ASRSystem = None):
    """
    Generates and stores the maximum example pairs for all samples.
    
    Args:
        asr_system: Specific ASR system to generate examples for. If None, generates for all available systems.
    """
    file_bases = [
        "aprocsa1554a",
        "aprocsa1713a",
        "aprocsa1731a",
        "aprocsa1738a",
        "aprocsa1833a",
        "aprocsa1944a"
    ]

    # Determine which ASR systems to process
    if asr_system:
        systems_to_process = [asr_system]
    else:
        systems_to_process = get_available_asr_systems()
        if not systems_to_process:
            print("❌ No ASR systems found with data. Please check your data/asr_raw/ directory structure.")
            return

    print(f"🎯 Processing {len(systems_to_process)} ASR system(s): {[s.value for s in systems_to_process]}")
    print("=" * 70)

    for system in systems_to_process:
        print(f"\n🔄 Processing ASR System: {system.value}")
        print("-" * 50)
        
        data_paths = get_data_paths(system)
        ground_truth_folder = data_paths["ground_truth"]
        
        # Create output folder structure for this ASR system
        output_examples_folder = os.path.join("data/examples_max", system.value.replace(" ", "_"))
        Path(output_examples_folder).mkdir(parents=True, exist_ok=True)
        
        print(f"📂 Output folder: {output_examples_folder}")
        
        total_files_processed = 0
        
        for file_base in file_bases:
            print(f"  📝 Processing: {file_base}...")
            
            truth_path = os.path.join(ground_truth_folder, f"{file_base}.txt")
            asr_path = get_asr_file_path(file_base, system)
            
            if not os.path.exists(truth_path):
                print(f"    ⚠️  Ground truth file not found, skipping: {truth_path}")
                continue
            if not os.path.exists(asr_path):
                print(f"    ⚠️  ASR file not found, skipping: {asr_path}")
                continue
                
            try:
                asr_example_sentences, gt_example_sentences = pick_all_unique_error_sentences_via_exhaustive(
                    truth_path,
                    asr_path
                )
                
                if not asr_example_sentences and not gt_example_sentences:
                    print(f"    ℹ️  No example pairs were generated for {file_base}.")

                output_data = {
                    "file_base": file_base,
                    "asr_system": system.value,
                    "asr_examples": asr_example_sentences,
                    "gt_examples": gt_example_sentences,
                    "count": len(asr_example_sentences)
                }
                
                # Create filename with ASR system identifier
                output_filename = f"{file_base}_{system.value.replace(' ', '_')}_max_examples.json"
                output_json_path = os.path.join(output_examples_folder, output_filename)
                
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                
                print(f"    ✅ Generated {len(asr_example_sentences)} example pairs")
                print(f"    💾 Saved to: {output_json_path}")
                total_files_processed += 1
                
            except Exception as e:
                print(f"    ❌ FAILED for {file_base}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n📊 {system.value} Summary: {total_files_processed}/{len(file_bases)} files processed")
        
    print("\n" + "=" * 70)
    print(f"🎉 Completed processing all ASR systems!")
    print(f"📁 Maximum example JSON files are stored in: data/examples_max/")


def generate_examples_for_system(system_name: str):
    """
    Generate examples for a specific ASR system by name.
    
    Args:
        system_name: Name of the ASR system (e.g., "Azure", "Whisper", etc.)
    """
    try:
        # Find the ASR system enum by value
        system = None
        for asr_sys in ASRSystem:
            if asr_sys.value.lower() == system_name.lower():
                system = asr_sys
                break
        
        if not system:
            available_systems = [s.value for s in ASRSystem]
            print(f"❌ Unknown ASR system: {system_name}")
            print(f"📋 Available systems: {available_systems}")
            return
        
        generate_and_store_max_examples(system)
        
    except Exception as e:
        print(f"❌ Error generating examples for {system_name}: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Generate for specific ASR system
        system_name = " ".join(sys.argv[1:])
        generate_examples_for_system(system_name)
    else:
        # Generate for all available ASR systems
        generate_and_store_max_examples() 