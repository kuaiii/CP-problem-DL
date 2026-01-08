# SDN Topology Resilience Analysis Framework

This project provides a framework for analyzing the resilience of Software-Defined Network (SDN) topologies under various attack scenarios and controller placement strategies.

## Project Structure

```text
codes/
├── data/
│   ├── raw/                 # Raw topology data (.gml files)
│   └── processed/           # Processed intermediate data
├── src/
│   ├── topology/            # Topology management
│   │   ├── generators.py    # Network generation & loading
│   │   ├── optimization.py  # Topology reconstruction (Genetic Algorithm)
│   │   └── reconstruction.py # Simple enhancement strategies
│   ├── controller/          # Controller management
│   │   ├── manager.py       # SDN Manager (deployment, state)
│   │   ├── strategies.py    # Controller placement strategies
│   │   └── rl_optimizer.py  # RL-based optimization
│   ├── simulation/          # Simulation modules
│   │   └── dismantling.py   # Attack simulation (node/edge removal)
│   ├── analysis/            # Analysis metrics
│   │   └── metrics.py       # Metrics calculation
│   ├── visualization/       # Visualization
│   │   ├── plots.py         # Network plotting
│   │   └── result_plotter.py # Result charts
│   ├── training/            # Training scripts
│   │   └── offline_rl.py    # Offline RL training
│   └── utils/               # Utilities
│       └── io.py            # I/O helpers
├── results/                 # Experiment results
├── main.py                  # Main entry point for single experiment
├── batch_run.py             # Batch execution script
├── requirements.txt         # Dependencies
└── README.md                # This file
```

## Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Single Experiment
```bash
python main.py --dataset GtsCe --attack random --batch_size 10
```
Arguments:
- `--dataset`: Name of the GML file in `data/raw` (without extension).
- `--synthetic`: Flag to use synthetic BA network instead of GML file.
- `--nodes`: Number of nodes for synthetic network.
- `--attack`: Attack mode (`random`, `target`, `hybrid`).
- `--batch_size`: Number of simulations to run.

### 3. Run Batch Experiments
```bash
python batch_run.py
```
This will run a predefined set of experiments on multiple datasets and attack modes.

## Output
Results are saved in the `results/` directory with the following structure:
`results/{dataset_name}/{attack_mode}/{metric_type}/{experiment_id}/`

Each experiment folder contains:
- `plot_avg.png`: Bar chart comparing average resilience of different strategies.
- `{attack_mode}{i}.png`: Line chart of resilience degradation for each simulation run.
- `statistics.csv`: Mean and standard deviation of collapse points.
- `r_statistics.csv`: Statistics for the robustness metric R.
- `x_values_results.csv`: Raw data points for network collapse.
- `r_values_results.csv`: Raw data points for robustness metric R.
