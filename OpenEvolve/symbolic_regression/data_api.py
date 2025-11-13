"""
Symbolic Regression Problem Generator

This module creates initial programs, evaluators, and configurations for symbolic regression tasks.
It processes multiple datasets in parallel and generates the necessary files for each problem.
"""

import os
import yaml
import numpy as np
import multiprocessing
import importlib.util
from typing import Dict, List, Tuple, Optional, Any

from bench.datamodules import get_datamodule


def load_secret(secrets_file: str = "secrets.yaml") -> Dict[str, Any]:
    """
    Load API keys and configuration from a secrets file.

    Args:
        secrets_file: Path to the YAML secrets file

    Returns:
        Dictionary containing secret configuration, empty dict if file not found
    """
    try:
        with open(secrets_file, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: Secrets file '{secrets_file}' not found.")
        return {}
    except Exception as e:
        print(f"Warning: Error loading secrets file '{secrets_file}': {e}")
        return {}


def extract_problem_data_from_initialized_dataset(
    initialized_dataset, problem_id: int
) -> Dict[str, Any]:
    """
    Extract data for a specific problem from an initialized dataset.

    Args:
        initialized_dataset: Pre-initialized and setup dataset object
        problem_id: Index of the problem to extract

    Returns:
        Dictionary containing problem data including train/test samples, symbols, and metadata
    """
    problem = initialized_dataset.problems[problem_id]
    gt_eq = problem.gt_equation
    samples = problem.samples

    data = {
        "train": problem.train_samples,
        "test": problem.test_samples,
        "ood_test": problem.ood_test_samples,
        "symbols": gt_eq.symbols,
        "symbol_descs": gt_eq.symbol_descs,
        "symbol_properties": gt_eq.symbol_properties,
        "expression": gt_eq.expression,
        "dataset_identifier": problem.dataset_identifier,
        "equation_idx": problem.equation_idx,
        "problem_type": problem.problem_type,
    }
    return data


def create_program(problem: Dict[str, Any]) -> str:
    """
    Create a Python script with a naive linear model for symbolic regression.

    The generated script contains a `func(x, params)` that computes predictions
    in a vectorized manner: x @ params. If no input features exist, it predicts
    a constant params[0].

    Args:
        problem: Dictionary containing problem data

    Returns:
        Path to the created program file
    """
    problem_dir = f'problems/{problem["dataset_identifier"]}/{problem["equation_idx"]}'

    symbols = problem["symbols"]
    properties = problem["symbol_properties"]
    descs = problem["symbol_descs"]

    input_vars = []
    input_vars_descs = []
    output_var = None
    output_var_desc = "N/A"

    for i, prop in enumerate(properties):
        if prop == "V":
            input_vars.append(symbols[i])
            input_vars_descs.append(descs[i])
        elif prop == "O":
            output_var = symbols[i]
            output_var_desc = descs[i]

    if not output_var:
        raise ValueError("No output variable ('O') found in symbol_properties.")

    x_mapping_comments = ["# Input variable mapping for x (columns of the input matrix):"]
    if not input_vars:
        x_mapping_comments.append("#   No input variables (x will be an (n_samples, 0) matrix).")
    else:
        for i, var_name in enumerate(input_vars):
            x_mapping_comments.append(f"#   x[:, {i}]: {var_name} ({input_vars_descs[i]})")
    x_mapping_str = "\n".join(x_mapping_comments)

    # Build function body
    num_features = len(input_vars)
    if num_features > 0:
        function_body = " + ".join([f"x[:, {i}] * params[{i}]" for i in range(num_features)])
    else:
        function_body = (
            "np.full(x.shape[0], params[0])  # Predicts a constant value for all samples"
        )

    model_num_params = 10

    # Build input variables description
    input_vars_desc_list = [f"{v} ({input_vars_descs[i]})" for i, v in enumerate(input_vars)]
    input_vars_desc_str = ", ".join(input_vars_desc_list) if input_vars else "None"

    program_content = f'''"""
Initial program: A naive linear model for symbolic regression.
This model predicts the output as a linear combination of input variables
or a constant if no input variables are present.
The function is designed for vectorized input (X matrix).

Target output variable: {output_var} ({output_var_desc})
Input variables (columns of x): {input_vars_desc_str}
"""
import numpy as np

{x_mapping_str}

# Parameters will be optimized by BFGS outside this function.
# Number of parameters expected by this model: {model_num_params}.
# Example initialization: params = np.random.rand({model_num_params})

# EVOLVE-BLOCK-START

def func(x, params):
    """
    Calculates the model output using a linear combination of input variables
    or a constant value if no input variables. Operates on a matrix of samples.

    Args:
        x (np.ndarray): A 2D numpy array of input variable values, shape (n_samples, n_features).
                        n_features is {num_features}.
                        If n_features is 0, x should be shape (n_samples, 0).
                        The order of columns in x must correspond to:
                        ({', '.join(input_vars) if input_vars else "None - x has 0 columns"}).
        params (np.ndarray): A 1D numpy array of parameters.
                             Expected length: {model_num_params}.

    Returns:
        np.ndarray: A 1D numpy array of predicted output values, shape (n_samples,).
    """
    result = {function_body}
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
'''

    os.makedirs(problem_dir, exist_ok=True)
    file_path = os.path.join(problem_dir, "initial_program.py")
    with open(file_path, "w") as f:
        f.write(program_content)

    return file_path


def create_evaluator(problem: Dict[str, Any]) -> str:
    """
    Create an evaluator script for the symbolic regression problem.

    The evaluator assesses model performance using BFGS optimization
    and computes various metrics including MSE and combined scores.

    Args:
        problem: Dictionary containing problem data

    Returns:
        Path to the created evaluator file
    """
    problem_dir = f'problems/{problem["dataset_identifier"]}/{problem["equation_idx"]}'
    os.makedirs(problem_dir, exist_ok=True)

    properties = problem["symbol_properties"]
    train_samples = np.asarray(problem["train"])
    
    test_samples = np.asarray(problem["test"]) if problem["test"] is not None else None
    ood_test_samples = np.asarray(problem["ood_test"]) if problem["ood_test"] is not None else None

    input_indices = [i for i, prop in enumerate(properties) if prop == "V"]
    output_indices = [i for i, prop in enumerate(properties) if prop == "O"]

    if not output_indices:
        raise ValueError("No output variable ('O') found in symbol_properties.")
    if len(output_indices) > 1:
        raise ValueError("Multiple output variables ('O') found. Evaluator supports single output.")
    output_index = output_indices[0]

    num_input_features = len(input_indices)
    
    X_train = train_samples[:, input_indices] if num_input_features > 0 else np.empty((len(train_samples), 0))
    y_train = train_samples[:, output_index]

    X_test, y_test = (None, None)
    if test_samples is not None:
        X_test = test_samples[:, input_indices] if num_input_features > 0 else np.empty((len(test_samples), 0))
        y_test = test_samples[:, output_index]

    X_ood_test, y_ood_test = (None, None)
    if ood_test_samples is not None:
        X_ood_test = ood_test_samples[:, input_indices] if num_input_features > 0 else np.empty((len(ood_test_samples), 0))
        y_ood_test = ood_test_samples[:, output_index]

    model_num_params_expected = 10

    # Save data files
    base_data_path = "./"
    x_train_path = os.path.join(base_data_path, problem_dir, "X_train_for_eval.npy")
    y_train_path = os.path.join(base_data_path, problem_dir, "y_train_for_eval.npy")
    np.save(x_train_path, X_train)
    np.save(y_train_path, y_train)

    if X_test is not None and y_test is not None:
        x_test_path = os.path.join(problem_dir, "X_test_for_eval.npy")
        y_test_path = os.path.join(problem_dir, "y_test_for_eval.npy")
        np.save(x_test_path, X_test)
        np.save(y_test_path, y_test)

    if X_ood_test is not None and y_ood_test is not None:
        x_ood_test_path = os.path.join(problem_dir, "X_ood_test_for_eval.npy")
        y_ood_test_path = os.path.join(problem_dir, "y_ood_test_for_eval.npy")
        np.save(x_ood_test_path, X_ood_test)
        np.save(y_ood_test_path, y_ood_test)

    evaluator_script_content = f'''"""
Evaluator for a symbolic regression model.
It assesses a model program based on its performance on training data.
The model's `func` is expected to take a matrix X of inputs.
"""
import os
import sys
import time
import importlib.util
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import concurrent.futures

# Expected number of input features for the model's func
NUM_INPUT_FEATURES_EXPECTED = {num_input_features}
# Expected number of parameters for the initial model
MODEL_NUM_PARAMS_EXPECTED = {model_num_params_expected}

# Paths to data
X_TRAIN_EVAL_PATH = r'{x_train_path}'
Y_TRAIN_EVAL_PATH = r'{y_train_path}'

def chamfer_distance(a, b):
    """Compute the Chamfer distance between two sets of points."""
    dists = cdist(a, b)
    return np.mean(np.min(dists, axis=1)) + np.mean(np.min(dists, axis=0))

def hausdorff_distance(a, b):
    """Compute the Hausdorff distance between two sets of points."""
    dists = cdist(a, b)
    return max(np.max(np.min(dists, axis=1)), np.max(np.min(dists, axis=0)))

def run_with_timeout(func, args=(), kwargs={{}}, timeout_seconds=5):
    """Execute a function with a timeout."""
    if timeout_seconds is None or timeout_seconds <= 0:
        return func(*args, **kwargs)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            func_name = getattr(func, '__name__', 'Unnamed function')
            raise TimeoutError(f"Function {{func_name}} timed out after {{timeout_seconds}} seconds")


def filter_and_convert_metrics(current_metrics_dict):
    """Filter and convert metrics to appropriate types."""
    filtered_dict = {{}}
    float_metric_keys = ['combined_score', 'negative_mse', 'chamfer', 'hausdorff']
    
    for key in float_metric_keys:
        if key in current_metrics_dict:
            value = current_metrics_dict[key]
            if value is None:
                continue
            if isinstance(value, (int, float, np.integer, np.floating, bool)):
                try:
                    filtered_dict[key] = float(value)
                except (ValueError, TypeError):
                    pass
    
    return filtered_dict


def objective_function(params, model_func, X_matrix, y_true_vector):
    """Objective function for scipy.optimize.minimize (calculates MSE)."""
    if not callable(model_func):
        return float('inf')
    
    try:
        predictions = model_func(X_matrix, params)
        if not isinstance(predictions, np.ndarray) or predictions.shape != y_true_vector.shape:
            return float('inf')
        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            return float('inf')
        return np.mean((predictions - y_true_vector)**2)
    except Exception:
        return float('inf')

def evaluate(program_path):
    """Evaluate a model program on the training data."""
    metrics = {{
        'can_run': 0.0,
        'negative_mse': -1e09,
        'raw_mse_train': float('inf'),
        'mse_train_score': 0.0,
        'chamfer': float('inf'),
        'hausdorff': float('inf'),
        'num_params': MODEL_NUM_PARAMS_EXPECTED,
        'combined_score': -1e09,
        'error_message': None,
        'optimization_success': False,
        'optimized_params': None
    }}
    
    # Load training data
    try:
        X_train = np.load(X_TRAIN_EVAL_PATH)
        y_train = np.load(Y_TRAIN_EVAL_PATH)
        
        if X_train.shape[1] != NUM_INPUT_FEATURES_EXPECTED:
            metrics['error_message'] = f"Loaded X_train has {{X_train.shape[1]}} features, expected {{NUM_INPUT_FEATURES_EXPECTED}}."
            return filter_and_convert_metrics(metrics)
        if X_train.shape[0] != y_train.shape[0]:
            metrics['error_message'] = f"X_train has {{X_train.shape[0]}} samples, y_train has {{y_train.shape[0]}}."
            return filter_and_convert_metrics(metrics)
    except Exception as e:
        metrics['error_message'] = f"Failed to load training data: {{str(e)}}"
        return filter_and_convert_metrics(metrics)
    
    func_to_eval = None
    try:
        spec = importlib.util.spec_from_file_location("model_program", program_path)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        metrics['can_run'] = 0.2
        
        if not hasattr(model_module, 'run_search') or not callable(model_module.run_search):
            metrics['error_message'] = "Model program missing callable 'run_search'."
            return filter_and_convert_metrics(metrics)
        
        func_to_eval = model_module.run_search()
        
        if not callable(func_to_eval):
            metrics['error_message'] = "'run_search' did not return a callable function."
            return filter_and_convert_metrics(metrics)
        
        # Test the function with dummy data
        dummy_x = np.random.rand(5, NUM_INPUT_FEATURES_EXPECTED)
        dummy_params = np.random.rand(MODEL_NUM_PARAMS_EXPECTED)
        pred_test = run_with_timeout(func_to_eval, args=(dummy_x, dummy_params), timeout_seconds=5)
        if not isinstance(pred_test, np.ndarray) or pred_test.shape != (5,):
            metrics['can_run'] = 0.5
            metrics['error_message'] = "Func test output shape mismatch."
            return filter_and_convert_metrics(metrics)
        metrics['can_run'] = 1.0
    except Exception as e:
        metrics['error_message'] = f"Failed to load or test model function: {{str(e)}}"
        return filter_and_convert_metrics(metrics)

    if metrics['can_run'] < 1.0:
        return filter_and_convert_metrics(metrics)
    
    # Optimize parameters
    initial_params = np.random.rand(MODEL_NUM_PARAMS_EXPECTED)
    try:
        opt_result = minimize(
            objective_function, initial_params,
            args=(func_to_eval, X_train, y_train), method='BFGS'
        )
        metrics['raw_mse_train'] = opt_result.fun if np.isfinite(opt_result.fun) else float('inf')
        metrics['optimization_success'] = opt_result.success
        optimized_params = opt_result.x if opt_result.success else initial_params
        metrics['optimized_params'] = optimized_params.tolist()
    except Exception as e:
        metrics['raw_mse_train'] = float('inf')
        metrics['error_message'] = f"Error during optimization: {{str(e)}}"

    # Calculate final scores and distances
    if np.isfinite(metrics['raw_mse_train']):
        metrics['negative_mse'] = -metrics['raw_mse_train']
        metrics['mse_train_score'] = -np.log10(metrics['raw_mse_train'] + 1e-9)
        metrics['combined_score'] = metrics['mse_train_score']

        try:
            if metrics['optimized_params'] is not None:
                train_predictions = func_to_eval(X_train, np.array(metrics['optimized_params']))
                true_points = np.column_stack((X_train, y_train))
                pred_points = np.column_stack((X_train, train_predictions))
                metrics['chamfer'] = chamfer_distance(true_points, pred_points)
                metrics['hausdorff'] = hausdorff_distance(true_points, pred_points)
        except Exception as e:
            if not metrics['error_message']: # Avoid overwriting a previous error
                metrics['error_message'] = f"Error during distance calculation: {{str(e)}}"

    return filter_and_convert_metrics(metrics)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python evaluator.py <path_to_model_program.py>")
        print("Please run the main script that calls create_program and create_evaluator first.")
        sys.exit(1)
    
    program_to_evaluate = sys.argv[1]
    if not os.path.exists(program_to_evaluate):
        print(f"Error: Program file '{{program_to_evaluate}}' not found.")
        sys.exit(1)
    
    print(f"Evaluating model: {{program_to_evaluate}}")
    print(f"Using NUM_INPUT_FEATURES_EXPECTED = {{NUM_INPUT_FEATURES_EXPECTED}}")
    print(f"Using MODEL_NUM_PARAMS_EXPECTED = {{MODEL_NUM_PARAMS_EXPECTED}}")
    print(f"Loading X_train from: {{X_TRAIN_EVAL_PATH}}")
    print(f"Loading y_train from: {{Y_TRAIN_EVAL_PATH}}")
    
    if not os.path.exists(X_TRAIN_EVAL_PATH):
        print(f"Error: X_train data file '{{X_TRAIN_EVAL_PATH}}' not found.")
        sys.exit(1)
    if not os.path.exists(Y_TRAIN_EVAL_PATH):
        print(f"Error: y_train data file '{{Y_TRAIN_EVAL_PATH}}' not found.")
        sys.exit(1)
    
    evaluation_results = evaluate(program_to_evaluate)
    print("\\nEvaluation Results:")
    for key, value in evaluation_results.items():
        if isinstance(value, float):
            print(f"  {{key}}: {{value:.4f}}")
        else:
            print(f"  {{key}}: {{value}}")
'''
    evaluator_file_path = os.path.join(problem_dir, "evaluator.py")
    with open(evaluator_file_path, "w") as f:
        f.write(evaluator_script_content)

    return evaluator_file_path


def create_config(problem: Dict[str, Any], problem_type: str) -> str:
    """
    Create a YAML configuration file for the symbolic regression task.

    Args:
        problem: Dictionary containing problem data

    Returns:
        Path to the created configuration file
    """
    problem_dir = f'problems/{problem["dataset_identifier"]}/{problem["equation_idx"]}'
    os.makedirs(problem_dir, exist_ok=True)
    config_file_path = os.path.join(problem_dir, "config.yaml")

    symbols = problem["symbols"]
    properties = problem["symbol_properties"]
    descs = problem["symbol_descs"]

    input_vars_list = []
    output_var_list = []

    for i, prop in enumerate(properties):
        if prop == "V":
            input_vars_list.append(f"{symbols[i]} ({descs[i]})")
        elif prop == "O":
            output_var_list.append(f"{symbols[i]} ({descs[i]})")

    input_vars_str = ", ".join(input_vars_list) if input_vars_list else "None"
    output_var_str = (
        ", ".join(output_var_list) if output_var_list else "None (Error: No output defined!)"
    )

    num_initial_params = 10

    system_message = (
        "Your task is to evolve a Python function `func(x, params)` that models a scientific process, "
        "considering the physical meaning and relationships of inputs, "
        "by predicting output variables based on input variables."
        "The function signature is:\\n\\n"
        "```python"
        "def func(x: np.ndarray, params: np.ndarray) -> np.ndarray:"
        "```"
        f"- `x` is a 2D NumPy array of shape `(n_samples, {len(input_vars_list)})`"
        f"- `params` is a 1D NumPy array of up to {num_initial_params} parameters"
        "- The function should return a 1D NumPy array of predictions with shape `(n_samples,)`"
        "**Current Problem:**"
        f"Model the {output_var_str} using the input features: {input_vars_str}\\n"
        f"Thus, `x` contains {len(input_vars_list)} columns: {input_vars_str}.\\n\\n"
        "The initial version of `func` is a simple linear model. Parameters in `params` will be optimized externally "
        "using the BFGS algorithm based on unseen training data.\\n\\n"
        "Your objective is to evolve `func` to improve predictive performance on unseen data. Aim for a balance between:\\n"
        "- **Accuracy**: Lower mean squared error (MSE) on training data\\n"
        "- **Simplicity**: Prefer concise, interpretable expressions\\n\\n"
        "Model performance (score = -log_10(mse)) will be evaluated on a held-out dataset. "
        "Ensure the model is free of potential numerical errors (e.g., log0, division by 0)."
    )

    secret = load_secret()
    config_data = {
        "# Configuration for Symbolic Regression Task": f"{problem['dataset_identifier']}/{problem['equation_idx']}",
        "max_iterations": 1000,
        "max_tokens": 4096,                    # Maximum tokens to generate
        "log_level": "INFO",
        "target_score": "combined_score",
        "checkpoint_interval": 10,
        "llm": {
            "primary_model": "gpt-4o-mini",
            "primary_model_weight": 0.8,
            "secondary_model": "gpt-4o-mini",
            "secondary_model_weight": 0.2,
            "api_base": "https://api.openai.com/v1",
        },
        "prompt": {
            "system_message": system_message,
            "num_top_programs": 4,
            "use_template_stochasticity": True,
        },
        "database": {
            "population_size": 70,
            "archive_size": 30,
            "num_islands": 10,
            "elite_selection_ratio": 0.3,
            "exploitation_ratio": 0.6,
        },
        "evaluator": {
            "timeout": 90,
            "cascade_evaluation": False,
            "cascade_thresholds": [1.0],
            "parallel_evaluations": 4,
            "use_llm_feedback": False,
        },
        "diff_based_evolution": True,
        "allow_full_rewrites": False,
    }

    class PreserveNewlinesDumper(yaml.SafeDumper):
        """Custom YAML dumper that preserves multi-line strings."""

        def represent_scalar(self, tag, value, style=None):
            if style is None and isinstance(value, str) and "\n" in value:
                style = "|"
            return super().represent_scalar(tag, value, style)

    with open(config_file_path, "w") as f:
        yaml.dump(
            config_data,
            f,
            Dumper=PreserveNewlinesDumper,
            default_flow_style=False,
            sort_keys=False,
            indent=2,
        )

    return config_file_path


def process_problem(initialized_dataset, problem_id: int, split_name: str) -> str:
    """
    Process a single problem using a pre-initialized dataset.

    Loads specific problem data, creates program, evaluator, and config.
    Skips processing if essential output files already exist.

    Args:
        initialized_dataset: Pre-initialized and setup dataset object
        problem_id: Index of the problem to process
        split_name: Name of the dataset split

    Returns:
        Status message indicating success, skip, or error
    """
    try:
        problem_data = extract_problem_data_from_initialized_dataset(
            initialized_dataset, problem_id
        )

        problem_type = problem_data.get("problem_type")
        if problem_type == "parametric":
            base_equation_idx = problem_data["equation_idx"]
            print(f"Processing PARAMETRIC problem: {base_equation_idx}")
            
            # Save the full test/ood data sets for the final evaluation
            parent_dir = os.path.join("problems", problem_data["dataset_identifier"], str(base_equation_idx))
            os.makedirs(parent_dir, exist_ok=True)
            if problem_data["test"] is not None:
                np.save(os.path.join(parent_dir, "test_data_eval.npy"), problem_data["test"])
            if problem_data["ood_test"] is not None:
                np.save(os.path.join(parent_dir, "ood_test_eval.npy"), problem_data["ood_test"])

            # Process each coordinate as a separate sub-problem
            for target_coord in ['x', 'y', 'z']:
                sub_problem_data = problem_data.copy()
                
                sub_problem_data["equation_idx"] = f"{base_equation_idx}/{target_coord}"

                # Update symbols and properties for the current target coordinate
                sub_problem_data["symbols"] = ['u', 'v', target_coord]
                sub_problem_data["symbol_properties"] = ['V', 'V', 'O']
                sub_problem_data["symbol_descs"] = ["parameter u", "parameter v", f"{target_coord} coordinate"]
                
                sub_problem_data["train"] = problem_data["train"][target_coord]
                
                if problem_data["test"] is not None and isinstance(problem_data["test"], dict):
                    sub_problem_data["test"] = problem_data["test"][target_coord]
                else:
                    sub_problem_data["test"] = None

                if problem_data["ood_test"] is not None and isinstance(problem_data["ood_test"], dict):
                    sub_problem_data["ood_test"] = problem_data["ood_test"][target_coord]
                else:
                    sub_problem_data["ood_test"] = None

                create_program(sub_problem_data)
                create_evaluator(sub_problem_data)
                create_config(sub_problem_data, problem_type="parametric")
            
            return f"Successfully processed PARAMETRIC problem: {base_equation_idx} (created x, y, z sub-problems)"

        else: # Handle standard, non-parametric problems
            dataset_identifier = problem_data["dataset_identifier"]
            equation_idx = problem_data["equation_idx"]
            
            problem_dir = os.path.join("problems", dataset_identifier, str(equation_idx))
            essential_files = [os.path.join(problem_dir, f) for f in
                               ["initial_program.py", "evaluator.py", "config.yaml"]]
            
            if all(os.path.exists(f) for f in essential_files):
                return f"Skipped (already processed): problem_id: {problem_id} for split: {split_name} ({dataset_identifier}/{equation_idx})"

            create_program(problem_data)
            create_evaluator(problem_data)
            create_config(problem_data, problem_type="standard")

            return f"Successfully processed problem_id: {problem_id} for split: {split_name} ({dataset_identifier}/{equation_idx})"

    except Exception as e:
        import traceback
        return f"Error processing problem_id {problem_id} for split {split_name}: {str(e)}\n{traceback.format_exc()}"

def main():
    """
    Main entry point for processing symbolic regression problems.

    Initializes datasets and processes problems in parallel using multiprocessing.
    """
    # Determine number of processes to use
    num_cores_available = os.cpu_count()
    num_processes = min(max(1, (num_cores_available - 1) if num_cores_available else 4), 24)

    print(f"Starting processing with {num_processes} processes...")

    # Define dataset splits and their problem counts
    splits_data = {
        "Parametric_Equations": 30,
        "Algebraic Manifolds of Higher Degree": 10,
        "Bio-Inspired_Morphological_Surfaces": 10,
        "Complex_Composite_Surfaces": 10,
        "Discrete_Symbolic_Grid_Surfaces": 10,
        "Hybrid_Multi-Modal_Symbolic_Surfaces": 10,
        "Non-Canonical_3D_Geometric_Surfaces": 11,
        "Piecewise_Regime_Surfaces": 10,
        "Procedural_Fractal_Surfaces": 10,
        "Symbolic-Numeric_Composite_Surfaces": 10,
        "Tensor_Field_Surfaces": 10,
        "Quantum_Inspired_Surfaces":10,
        "Nonlinear_Dynamical_System_Surfaces":9,
        "Stochastic_Process_Surfaces":10,
        "Surrogate_Distilled_Symbolic_Approximations":9
    }

    all_tasks = []

    # Initialize datasets and prepare tasks
    for split_name, num_problems in splits_data.items():
        print(f"\nInitializing dataset for split: {split_name}...")
        dataset_root_folder = f"dataset/{split_name}"

        try:
            # Initialize and setup dataset once per split
            initialized_dataset = get_datamodule(split_name, dataset_root_folder)
            initialized_dataset.setup()
            print(f"Dataset for {split_name} initialized and setup complete.")

            # Prepare tasks for this split
            print(f"Preparing tasks for split: {split_name} ({num_problems} problems)")
            for problem_id in range(num_problems):
                all_tasks.append((initialized_dataset, problem_id, split_name))

        except Exception as e:
            print(
                f"ERROR: Could not initialize or setup dataset for split {split_name}. Skipping this split."
            )
            print(f"Details: {e}")
            import traceback

            traceback.print_exc()
            continue

    if not all_tasks:
        print(
            "No tasks to process. This could be due to errors in dataset initialization. Exiting."
        )
        return

    print(f"\nTotal tasks to process across all successfully initialized splits: {len(all_tasks)}")

    # Process tasks in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(process_problem, all_tasks)

    # Print results summary
    print("\n--- Processing Complete ---")
    success_count = 0
    skipped_count = 0
    error_count = 0

    for result in results:
        print(result)
        if "Successfully processed" in result:
            success_count += 1
        elif "Skipped" in result:
            skipped_count += 1
        elif "Error processing" in result:
            error_count += 1

    print(f"\nSummary: {success_count} successful, {skipped_count} skipped, {error_count} errors.")
    print("\nAll tasks finished.")


if __name__ == "__main__":
    main()