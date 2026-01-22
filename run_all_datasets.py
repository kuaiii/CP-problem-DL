# -*- coding: utf-8 -*-
import os
import time
import subprocess
import argparse
import logging
from src.utils.logger import setup_logger, get_logger

logger = get_logger(__name__)

def get_datasets(data_dirs):
    """
    Get a list of dataset names (without extension) from the GML files in the directories.
    
    Args:
        data_dirs (list): List of directories to search for GML files
        
    Returns:
        list: Sorted list of unique dataset names
    """
    datasets = set()
    
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            logger.warning(f"Directory {data_dir} does not exist, skipping.")
            continue
            
        for filename in os.listdir(data_dir):
            if filename.endswith(".gml"):
                dataset_name = os.path.splitext(filename)[0]
                datasets.add(dataset_name)
    
    return sorted(list(datasets))

def run_simulation(dataset, attack_mode, batch_size, rate=0.1, synthetic=False, nodes=100, debug=False):
    """
    Run the simulation for a specific dataset and attack mode.
    
    Args:
        dataset (str): Dataset name
        attack_mode (str): Attack mode
        batch_size (int): Number of simulation runs
        rate (float): Controller placement rate
        synthetic (bool): Whether to use synthetic network
        nodes (int): Number of nodes for synthetic network
        debug (bool): Enable debug logging
        
    Returns:
        bool: True if successful, False otherwise
    """
    cmd = ["python", "main.py"]
    
    if synthetic:
        cmd.extend(["--synthetic", "--nodes", str(nodes)])
        # For synthetic, dataset name is generated automatically
        logger.info(f"Starting synthetic simulation: BA_{nodes}, Attack={attack_mode}, Batch={batch_size}")
    else:
        cmd.extend(["--dataset", dataset])
        logger.info(f"Starting simulation: Dataset={dataset}, Attack={attack_mode}, Batch={batch_size}")
    
    cmd.extend([
        "--attack", attack_mode,
        "--batch_size", str(batch_size),
        "--rate", str(rate)
    ])
    
    if debug:
        cmd.append("--debug")
    
    logger.debug(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the command and wait for it to complete
        result = subprocess.run(cmd, check=True, capture_output=False)
        logger.info(f"Finished: {dataset if not synthetic else f'BA_{nodes}'} ({attack_mode})")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {dataset if not synthetic else f'BA_{nodes}'} ({attack_mode}): {e}")
        return False
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user.")
        return False

def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Batch run simulations for multiple datasets')
    parser.add_argument('--data_dirs', type=str, nargs='+', 
                       default=['data/testdata', 'data/raw'],
                       help='Directories to search for GML files (default: data/testdata data/raw)')
    parser.add_argument('--attack_modes', type=str, nargs='+',
                       default=['random', 'degree'],
                       choices=['hybrid', 'random', 'target', 'degree', 'pagerank', 'betweenness', 'eigenvector', 'bruteforce'],
                       help='Attack modes to run (default: random degree)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Number of simulation runs per dataset (default: 1)')
    parser.add_argument('--rate', type=float, default=0.1,
                       help='Controller placement rate (default: 0.1)')
    parser.add_argument('--synthetic', action='store_true',
                       help='Run synthetic BA network simulations')
    parser.add_argument('--synthetic_nodes', type=int, nargs='+', default=[100],
                       help='Number of nodes for synthetic networks (default: [100])')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                       help='Specific datasets to run (default: all found datasets)')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip datasets that already have results')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    return parser.parse_args()

def check_existing_results(dataset, attack_mode, base_dir='results'):
    """
    Check if results already exist for a dataset and attack mode.
    
    Args:
        dataset (str): Dataset name
        attack_mode (str): Attack mode
        base_dir (str): Base results directory
        
    Returns:
        bool: True if results exist, False otherwise
    """
    result_dir = os.path.join(base_dir, dataset, attack_mode)
    if not os.path.exists(result_dir):
        return False
    
    # Check if there are any experiment directories
    subdirs = [d for d in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, d))]
    numeric_subdirs = [d for d in subdirs if d.isdigit()]
    
    if not numeric_subdirs:
        return False
    
    # Check if the latest experiment has results
    latest_exp = max(numeric_subdirs, key=int)
    exp_dir = os.path.join(result_dir, latest_exp)
    
    # Check if any metric directories exist and have files
    metric_types = ['lcc', 'csa', 'cce', 'wcp']
    for metric_type in metric_types:
        metric_dir = os.path.join(exp_dir, metric_type)
        if os.path.exists(metric_dir):
            # Check if subdirectories exist
            subdirs = ['curves', 'collapse_point', 'area']
            for subdir in subdirs:
                subdir_path = os.path.join(metric_dir, subdir)
                if os.path.exists(subdir_path) and os.listdir(subdir_path):
                    return True
    
    return False

def main():
    args = parse_arguments()
    
    # Setup logger
    setup_logger(log_dir="logs", log_filename="batch_run.log", 
                level=logging.DEBUG if args.debug else logging.INFO)
    
    # Get datasets
    if args.synthetic:
        # For synthetic networks, use node counts as "datasets"
        datasets = [f"BA_{nodes}" for nodes in args.synthetic_nodes]
        logger.info(f"Running synthetic networks: {datasets}")
    else:
        datasets = get_datasets(args.data_dirs)
        if not datasets:
            logger.warning("No datasets found.")
            return
        
        # Filter to specific datasets if provided
        if args.datasets:
            datasets = [d for d in datasets if d in args.datasets]
            if not datasets:
                logger.warning(f"None of the specified datasets found: {args.datasets}")
                return
        
        logger.info(f"Found {len(datasets)} datasets: {', '.join(datasets)}")
    
    # Calculate total tasks
    total_tasks = len(datasets) * len(args.attack_modes)
    completed_tasks = 0
    skipped_tasks = 0
    failed_tasks = 0
    
    start_time = time.time()
    
    logger.info(f"Starting batch run: {total_tasks} tasks")
    logger.info(f"  Datasets: {len(datasets)}")
    logger.info(f"  Attack modes: {', '.join(args.attack_modes)}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Controller rate: {args.rate}")
    
    for dataset in datasets:
        for attack in args.attack_modes:
            # Check if we should skip existing results
            if args.skip_existing and not args.synthetic:
                if check_existing_results(dataset, attack):
                    logger.info(f"Skipping {dataset} ({attack}): results already exist")
                    skipped_tasks += 1
                    completed_tasks += 1
                    continue
            
            # Determine if this is a synthetic network
            is_synthetic = args.synthetic or dataset.startswith("BA_")
            nodes = int(dataset.split("_")[1]) if dataset.startswith("BA_") else 100
            
            success = run_simulation(
                dataset=dataset,
                attack_mode=attack,
                batch_size=args.batch_size,
                rate=args.rate,
                synthetic=is_synthetic,
                nodes=nodes,
                debug=args.debug
            )
            
            if not success:
                failed_tasks += 1
                logger.warning(f"Task failed: {dataset} ({attack})")
            else:
                completed_tasks += 1
            
            logger.info(f"Progress: {completed_tasks}/{total_tasks} completed, "
                       f"{skipped_tasks} skipped, {failed_tasks} failed")
    
    elapsed_time = time.time() - start_time
    logger.info("="*60)
    logger.info(f"Batch run completed!")
    logger.info(f"  Total time: {elapsed_time/60:.2f} minutes ({elapsed_time:.2f} seconds)")
    logger.info(f"  Completed: {completed_tasks}/{total_tasks}")
    logger.info(f"  Skipped: {skipped_tasks}")
    logger.info(f"  Failed: {failed_tasks}")
    logger.info("="*60)

if __name__ == "__main__":
    main()
