from models.azure_gpt import postprocess_4_1_4
from evaluators.evaluator import evaluate_with_example_pairs
from utils.file_utils import create_or_replace_experiment_folder, list_available_experiments
from utils.example_generator import select_strategy_interactively, get_available_strategies, ExampleStrategy
import os

def main():
    """Main entry point for running experiments."""
    NUM_RUNS_PER_EXPERIMENT = 5

    print("STEP 1: Select Example Generation Strategies")
    print("=" * 50)
    print("Available strategies:")
    strategies = get_available_strategies()
    for i, strategy in enumerate(strategies, 1):
        print(f"  {i}. {strategy.replace('_', ' ').title()}")
    
    print("\nStrategy selection options:")
    print("  'all' - Run all strategies")
    print("  '1,2,3' - Run specific strategies by number")
    print("  'data_driven' - Run specific strategy by name")
    
    # Let user select multiple strategies
    strategy_input = input("\nEnter strategy selection: ").strip().lower()
    
    selected_strategies = []
    if strategy_input == 'all':
        selected_strategies = [ExampleStrategy(strategy) for strategy in strategies]
        print(f"Selected ALL {len(selected_strategies)} strategies")
    elif ',' in strategy_input:
        # Handle comma-separated numbers or names
        selections = [s.strip() for s in strategy_input.split(',')]
        for selection in selections:
            try:
                if selection.isdigit():
                    # Number selection
                    idx = int(selection) - 1
                    if 0 <= idx < len(strategies):
                        selected_strategies.append(ExampleStrategy(strategies[idx]))
                else:
                    # Name selection
                    selected_strategies.append(ExampleStrategy(selection))
            except ValueError:
                print(f"Invalid strategy: {selection}")
        print(f"Selected {len(selected_strategies)} strategies")
    else:
        # Single selection
        try:
            if strategy_input.isdigit():
                idx = int(strategy_input) - 1
                if 0 <= idx < len(strategies):
                    selected_strategies = [ExampleStrategy(strategies[idx])]
            else:
                selected_strategies = [ExampleStrategy(strategy_input)]
            print(f"Selected strategy: {selected_strategies[0].value}")
        except ValueError:
            print(f"Invalid strategy: {strategy_input}")
            print("Using default Data Driven strategy")
            selected_strategies = [ExampleStrategy.DATA_DRIVEN]
    
    if not selected_strategies:
        print("No valid strategies selected. Using default Data Driven strategy")
        selected_strategies = [ExampleStrategy.DATA_DRIVEN]
    
    print("\nSTEP 2: Select Experiments to Run")
    print("=" * 50)
    print("Available experiments:")
    experiments = list_available_experiments()
    for exp_id, description in experiments.items():
        print(f"  {exp_id}: {description}")
    
    print("\nSelect experiments to run (comma-separated, e.g., '4_1_4'):")
    selected_input = input("Enter experiment IDs: ").strip()
    if not selected_input:
        print("No experiments selected. Exiting.")
        return
    selected_ids = selected_input.split(',')
    
    experiment_functions = {
        "414": postprocess_4_1_4,
    }
    
    num_sentences_list = [2, 4, 6, 8, 10]
    
    print("\n" + "=" * 50)
    print(f"CONFIG: Each selected experiment will run {NUM_RUNS_PER_EXPERIMENT} time(s).")
    print(f"STRATEGIES: Running {len(selected_strategies)} strategies:")
    for strategy in selected_strategies:
        print(f"  - {strategy.value.replace('_', ' ').title()}")
    print("RUNNING EXPERIMENTS")
    print("=" * 50)
    
    for strategy_idx, current_strategy in enumerate(selected_strategies, 1):
        print(f"\nSTRATEGY {strategy_idx}/{len(selected_strategies)}: {current_strategy.value.replace('_', ' ').title()}")
        print("=" * 60)
        
        for base_exp_id in selected_ids:
            base_exp_id = base_exp_id.strip()
            if not base_exp_id:
                continue
                
            if base_exp_id not in experiment_functions:
                print(f"Warning: Experiment ID '{base_exp_id}' not found, skipping...")
                continue
            
            print(f"\nProcessing base experiment: {base_exp_id} with {current_strategy.value}")
            
            for run_num in range(NUM_RUNS_PER_EXPERIMENT):
                current_run_number = run_num + 1
                current_exp_id_for_folder = f"{base_exp_id}_{current_strategy.value}_run{current_run_number}"

                print(f"  Starting Run {current_run_number}/{NUM_RUNS_PER_EXPERIMENT} for {base_exp_id}")
                
                experiment_folder = create_or_replace_experiment_folder(current_exp_id_for_folder)
                
                for num_sentences in num_sentences_list:
                    print(f"    Testing {base_exp_id} (Run {current_run_number}) with {num_sentences} example sentences...")
                    try:
                        evaluate_with_example_pairs(
                            experiment_functions[base_exp_id], 
                            experiment_folder, 
                            num_sentences,
                            strategy=current_strategy
                        )
                    except Exception as e:
                        print(f"    Test failed for {base_exp_id} (Run {current_run_number}) with {num_sentences} sentences: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                print(f"  Completed Run {current_run_number}/{NUM_RUNS_PER_EXPERIMENT} for {base_exp_id}")
        
        print(f"\nCompleted all experiments for strategy: {current_strategy.value.replace('_', ' ').title()}")
        print(f"Progress: {strategy_idx}/{len(selected_strategies)} strategies completed")
    
    print("\nALL EXPERIMENTS COMPLETED!")
    print("=" * 50)
    print(f"Ran {len(selected_strategies)} strategies")
    print(f"Ran {len([id for id in selected_ids if id.strip()])} experiments")
    print(f"Each experiment ran {NUM_RUNS_PER_EXPERIMENT} time(s)")
    print("\nCheck the 'outputs' directory for results!")

if __name__ == "__main__":
    main() 