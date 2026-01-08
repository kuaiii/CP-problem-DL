import os
import time
import subprocess
import logging
from src.utils.logger import setup_logger, get_logger

logger = get_logger(__name__)

def get_datasets(data_dir):
    """
    Get a list of dataset names (without extension) from the GML files in the directory.
    """
    datasets = []
    if not os.path.exists(data_dir):
        logger.error(f"Error: Directory {data_dir} does not exist.")
        return datasets

    for filename in os.listdir(data_dir):
        if filename.endswith(".gml"):
            dataset_name = os.path.splitext(filename)[0]
            datasets.append(dataset_name)     
    return sorted(datasets)

def run_simulation(dataset, attack_mode, batch_size):
    """
    Run the simulation for a specific dataset and attack mode.
    """
    cmd = [
        "python", "main.py",
        "--dataset", dataset,
        "--attack", attack_mode,
        "--batch_size", str(batch_size)
    ]
    
    logger.info(f"Starting simulation: Dataset={dataset}, Attack={attack_mode}, Batch={batch_size}")
    logger.debug(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the command and wait for it to complete
        subprocess.run(cmd, check=True)
        logger.info(f"Finished: {dataset} ({attack_mode})")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {dataset} ({attack_mode}): {e}")
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user.")
        return False
    return True

def main():
    setup_logger(log_dir="logs", log_filename="batch_run.log")
    data_dir = os.path.join("data", "testdata")
    datasets = get_datasets(data_dir)
    
    if not datasets:
        logger.warning("No datasets found.")
        return

    logger.info(f"Found {len(datasets)} datasets: {', '.join(datasets)}")
    
    batch_size = 1
    attack_modes = ["random", "degree"]
    
    total_tasks = len(datasets) * len(attack_modes)
    completed_tasks = 0
    
    start_time = time.time()
    
    for dataset in datasets:
        for attack in attack_modes:
            success = run_simulation(dataset, attack, batch_size)
            if not success:
                logger.info("Stopping execution.")
                return
            completed_tasks += 1
            logger.info(f"Progress: {completed_tasks}/{total_tasks} tasks completed.")
            
    elapsed_time = time.time() - start_time
    logger.info(f"All simulations completed in {elapsed_time/60:.2f} minutes.")

if __name__ == "__main__":
    main()
