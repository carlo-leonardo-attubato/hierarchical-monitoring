import os
import json
import pickle
import datetime
from pathlib import Path
import matplotlib.pyplot as plt


def create_experiment_dir(experiment_name, base_dir="experiments"):
    """Create a timestamped directory for an experiment."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_experiment_config(exp_dir, config):
    """Save experiment configuration as JSON."""
    config_path = exp_dir / "config.json"

    # Convert any non-serializable objects to strings
    serializable_config = {}
    for key, value in config.items():
        try:
            json.dumps(value)
            serializable_config[key] = value
        except (TypeError, ValueError):
            serializable_config[key] = str(value)

    with open(config_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)
    return config_path


def save_experiment_results(exp_dir, results):
    """Save experiment results as JSON."""
    results_path = exp_dir / "results.json"

    # Convert any non-serializable objects to strings
    serializable_results = {}
    for key, value in results.items():
        try:
            json.dumps(value)
            serializable_results[key] = value
        except (TypeError, ValueError):
            serializable_results[key] = str(value)

    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    return results_path


def save_model(exp_dir, model, model_name="model"):
    """Save trained model."""
    model_path = exp_dir / f"{model_name}.pth"
    import torch
    torch.save(model.state_dict(), model_path)
    return model_path


def save_plot(exp_dir, plot_name, dpi=300, bbox_inches='tight'):
    """Save current matplotlib plot."""
    plot_path = exp_dir / f"{plot_name}.png"
    plt.savefig(plot_path, dpi=dpi, bbox_inches=bbox_inches)
    return plot_path


def save_data(exp_dir, data, filename):
    """Save data using pickle."""
    data_path = exp_dir / f"{filename}.pkl"
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)
    return data_path


def create_experiment_summary(exp_dir, experiment_name, description=""):
    """Create a README file for the experiment."""
    readme_path = exp_dir / "README.md"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    readme_content = f"""# {experiment_name}

**Timestamp**: {timestamp}
**Directory**: {exp_dir}

## Description
{description}

## Files Generated
- `config.json`: Experiment configuration parameters
- `results.json`: Final experiment results and metrics
- `*.png`: Generated plots and visualizations
- `*.pth`: Saved model weights
- `*.pkl`: Saved data objects

## Notes
{description}
"""

    with open(readme_path, 'w') as f:
        f.write(readme_content)
    return readme_path


class ExperimentLogger:
    """Context manager for experiment logging."""

    def __init__(self, experiment_name, description="", config=None):
        self.experiment_name = experiment_name
        self.description = description
        self.config = config or {}
        self.results = {}
        self.exp_dir = None

    def __enter__(self):
        self.exp_dir = create_experiment_dir(self.experiment_name)
        print(f"📁 Experiment directory: {self.exp_dir}")

        # Save config
        if self.config:
            save_experiment_config(self.exp_dir, self.config)

        # Create README
        create_experiment_summary(self.exp_dir, self.experiment_name, self.description)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Save final results
        if self.results:
            save_experiment_results(self.exp_dir, self.results)

        print(f"✅ Experiment saved to: {self.exp_dir}")

    def save_plot(self, plot_name, **kwargs):
        """Save current plot."""
        return save_plot(self.exp_dir, plot_name, **kwargs)

    def save_model(self, model, model_name="model"):
        """Save model."""
        return save_model(self.exp_dir, model, model_name)

    def save_data(self, data, filename):
        """Save data."""
        return save_data(self.exp_dir, data, filename)

    def log_result(self, key, value):
        """Log a result."""
        self.results[key] = value

    def log_results(self, results_dict):
        """Log multiple results."""
        self.results.update(results_dict)