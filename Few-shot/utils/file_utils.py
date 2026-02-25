import os
import shutil
from datetime import datetime
from enum import Enum
from typing import Dict, Optional

class ASRSystem(Enum):
    """Enumeration of available ASR systems"""
    AZURE_SPEECH = "Azure"
    ASSEMBLYAI = "AssemblyAI"
    WHISPER = "Whisper"
    GEMINI = "Gemini"
    AWS = "AWS"
    DEEPGRAM = "Deepgram"
    ELEVENLABS = "ElevenLabs"
    GCP = "GCP"
    GLADIA = "Gladia"
    SPEECHMATICS = "Speechmatics"

def create_or_replace_experiment_folder(exp_id: str, asr_system: ASRSystem = ASRSystem.AZURE_SPEECH) -> str:
    """
    Create or replace experiment folder based on experiment ID and ASR system.
    If folder exists, replace it.
    If doesn't exist, create new.
    
    Args:
        exp_id: e.g., "4_1_4"
        asr_system: The ASR system being used for this experiment
    
    Returns:
        Path to the experiment folder
    """
    # Create folder structure: outputs/[ASR_System]/[strategy]/[exp_id]
    folder_path = os.path.join("outputs", asr_system.value, exp_id)
    
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Replaced existing folder: {folder_path}")
    
    os.makedirs(folder_path, exist_ok=True)
    print(f"Created new folder: {folder_path}")
    
    return folder_path

def get_data_paths(asr_system: ASRSystem = ASRSystem.AZURE_SPEECH) -> Dict[str, str]:
    """
    Get paths to data folders based on the selected ASR system
    
    Args:
        asr_system: The ASR system to get paths for
        
    Returns:
        Dictionary with paths for ground_truth, asr_raw, and audio_files
    """
    return {
        "ground_truth": "data/ground_truth",
        "asr_raw": os.path.join("data/asr_raw", asr_system.value),
        "audio_files": "data/audio_files"
    }

def get_asr_file_path(file_base: str, asr_system: ASRSystem = ASRSystem.AZURE_SPEECH) -> str:
    """
    Get the correct ASR file path for a given file base and ASR system
    
    Args:
        file_base: Base name like "aprocsa1554a"
        asr_system: The ASR system
        
    Returns:
        Full path to the ASR file
    """
    data_paths = get_data_paths(asr_system)
    return os.path.join(data_paths["asr_raw"], f"{file_base} {asr_system.value}.txt")

def get_available_asr_systems() -> list[ASRSystem]:
    """
    Get list of ASR systems that have data available
    
    Returns:
        List of available ASR systems
    """
    available_systems = []
    base_asr_path = "data/asr_raw"
    
    if not os.path.exists(base_asr_path):
        return available_systems
    
    for system in ASRSystem:
        system_path = os.path.join(base_asr_path, system.value)
        if os.path.exists(system_path) and os.listdir(system_path):
            available_systems.append(system)
    
    return available_systems

def list_available_experiments():
    """
    List all available experiment functions from models
    """
    # This will be populated when we create the model files
    experiments = {
        "4_1_4": "Azure GPT-4.1 - Technique 4",
    }
    return experiments

def get_experiment_output_path(exp_id: str, asr_system: ASRSystem, strategy: str) -> str:
    """
    Get the output path for an experiment with ASR system and strategy
    
    Args:
        exp_id: Experiment ID like "4_1_1_run1"
        asr_system: The ASR system used
        strategy: The strategy used (e.g., "data_driven", "exhaustive_phoneme")
        
    Returns:
        Full path to the experiment output directory
    """
    return os.path.join("outputs", asr_system.value, strategy, exp_id)

def list_output_experiments(asr_system: Optional[ASRSystem] = None) -> Dict[str, list]:
    """
    List all available experiment outputs, optionally filtered by ASR system
    
    Args:
        asr_system: Optional ASR system to filter by
        
    Returns:
        Dictionary mapping ASR systems to their available experiments
    """
    outputs_base = "outputs"
    experiments = {}
    
    if not os.path.exists(outputs_base):
        return experiments
    
    # If specific ASR system requested, only check that one
    systems_to_check = [asr_system] if asr_system else ASRSystem
    
    for system in systems_to_check:
        if system is None:
            continue
            
        system_path = os.path.join(outputs_base, system.value)
        if not os.path.exists(system_path):
            continue
            
        system_experiments = []
        
        # List strategies under this ASR system
        for strategy in os.listdir(system_path):
            strategy_path = os.path.join(system_path, strategy)
            if not os.path.isdir(strategy_path):
                continue
                
            # List experiment runs under this strategy
            for exp_run in os.listdir(strategy_path):
                exp_run_path = os.path.join(strategy_path, exp_run)
                if os.path.isdir(exp_run_path):
                    system_experiments.append({
                        'strategy': strategy,
                        'experiment': exp_run,
                        'path': exp_run_path
                    })
        
        if system_experiments:
            experiments[system.value] = system_experiments
    
    return experiments 