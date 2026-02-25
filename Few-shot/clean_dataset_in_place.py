#!/usr/bin/env python3
"""
Cleans all text files in data/asr_raw/ and data/ground_truth/ in-place.
Applies jiwer.RemovePunctuation() and newline replacement to each file,
OVERWRITING the original content.

Now supports the new ASR system directory structure with multiple ASR systems.

!!! WARNING !!!
!!! THIS SCRIPT IS DESTRUCTIVE AND WILL MODIFY YOUR ORIGINAL DATA FILES. !!!
!!! PLEASE BACK UP YOUR data/asr_raw/ AND data/ground_truth/ DIRECTORIES BEFORE RUNNING. !!!
"""

import os
import jiwer # You might need to pip install jiwer if not already installed
from utils.file_utils import get_data_paths, ASRSystem, get_available_asr_systems

def clean_text_files_in_place(directory_path: str, file_pattern: str = "*.txt"):
    """
    Reads all text files in a directory, cleans their content, and overwrites them.
    
    Args:
        directory_path: Path to the directory to clean
        file_pattern: Pattern to match files (e.g., "*.txt", "*Azure.txt")
    """
    print(f"📂 Processing files in: {directory_path}")
    
    if not os.path.exists(directory_path):
        print(f"   ⚠️  Directory does not exist: {directory_path}")
        return
    
    file_count = 0
    cleaned_count = 0
    
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            file_count += 1
            
            try:
                print(f"  🧹 Cleaning {filename}...")
                with open(file_path, 'r', encoding='utf-8') as f:
                    original_content = f.read()
                
                # Apply the specific cleaning
                cleaned_content = jiwer.RemovePunctuation()(original_content)
                cleaned_content = cleaned_content.replace("\n", " ") # Replace all newlines with a space
                cleaned_content = cleaned_content.replace("\n\n", " ") # Just in case, though the above should cover it
                # Remove leading/trailing whitespace from the whole content and ensure single spaces
                cleaned_content = " ".join(cleaned_content.split()).strip()

                if original_content.strip() != cleaned_content: # Check if content actually changed to avoid unnecessary writes
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(cleaned_content)
                    print(f"    ✅ Overwritten with cleaned content.")
                    cleaned_count += 1
                else:
                    print(f"    ℹ️  Content already clean or cleaning resulted in no change.")

            except Exception as e:
                print(f"    ❌ FAILED to clean {filename}: {e}")
    
    print(f"📊 Finished processing {directory_path}: {cleaned_count}/{file_count} files were modified.")


def clean_asr_system_data(asr_system: ASRSystem):
    """
    Clean data for a specific ASR system.
    
    Args:
        asr_system: The ASR system to clean data for
    """
    print(f"\n🎯 Cleaning data for ASR system: {asr_system.value}")
    print("-" * 50)
    
    data_paths = get_data_paths(asr_system)
    asr_path = data_paths["asr_raw"]
    
    clean_text_files_in_place(asr_path)


def clean_all_asr_systems():
    """Clean data for all available ASR systems."""
    available_systems = get_available_asr_systems()
    
    if not available_systems:
        print("❌ No ASR systems found with data.")
        return
    
    print(f"🎯 Found {len(available_systems)} ASR system(s) with data:")
    for system in available_systems:
        print(f"   • {system.value}")
    
    for system in available_systems:
        clean_asr_system_data(system)


def clean_ground_truth_data():
    """Clean ground truth data (same for all ASR systems)."""
    print(f"\n🎯 Cleaning Ground Truth data")
    print("-" * 50)
    
    # Get ground truth path (same regardless of ASR system)
    data_paths = get_data_paths(ASRSystem.AZURE_SPEECH)  # Use any system, GT path is the same
    ground_truth_path = data_paths["ground_truth"]
    
    clean_text_files_in_place(ground_truth_path)


if __name__ == "__main__":
    print("⚠️  WARNING ⚠️")
    print("This script will OVERWRITE original files in your ASR and Ground Truth directories.")
    print("It will apply jiwer.RemovePunctuation() and normalize newlines.")
    print("PLEASE ENSURE YOU HAVE A BACKUP of data/asr_raw/ and data/ground_truth/ before proceeding.")
    print()
    
    # Show available ASR systems
    available_systems = get_available_asr_systems()
    if available_systems:
        print("📋 Available ASR systems:")
        for i, system in enumerate(available_systems, 1):
            print(f"   {i}. {system.value}")
        print(f"   {len(available_systems) + 1}. All systems")
        print(f"   {len(available_systems) + 2}. Ground truth only")
    else:
        print("❌ No ASR systems found with data.")
        exit(1)
    
    print()
    confirm = input("Type 'YES_CLEAN_MY_DATA' to confirm you want to proceed: ")
    
    if confirm == "YES_CLEAN_MY_DATA":
        print("\n🚀 Proceeding with in-place cleaning...")
        
        # Ask which systems to clean
        choice = input(f"\nWhich data to clean? (1-{len(available_systems) + 2}) or 'all': ")
        
        if choice.lower() == 'all' or choice == str(len(available_systems) + 1):
            # Clean all ASR systems and ground truth
            clean_all_asr_systems()
            clean_ground_truth_data()
            
        elif choice == str(len(available_systems) + 2):
            # Ground truth only
            clean_ground_truth_data()
            
        elif choice.isdigit() and 1 <= int(choice) <= len(available_systems):
            # Specific ASR system
            system_index = int(choice) - 1
            selected_system = available_systems[system_index]
            clean_asr_system_data(selected_system)
            
            # Ask if they also want to clean ground truth
            gt_choice = input("\nAlso clean ground truth data? (y/N): ")
            if gt_choice.lower() in ['y', 'yes']:
                clean_ground_truth_data()
        else:
            print("❌ Invalid choice")
            exit(1)
        
        print("\n🎉 In-place cleaning process completed.")
        print("\n📋 IMPORTANT NEXT STEPS:")
        print("1. Re-run 'python utils/generate_max_examples.py' to update your JSON example files based on this cleaned data.")
        print("2. Review changes in evaluators/evaluator.py (to remove redundant cleaning steps before remove_specific_sentences).")
        print("3. Test your experiments with the new cleaned data.")
    else:
        print("❌ Operation cancelled. Your data has not been modified.") 