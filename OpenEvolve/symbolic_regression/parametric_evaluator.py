import sys
import json
import importlib.util
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import time
import re

class NumpyFloatJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyFloatJSONEncoder, self).default(obj)


def load_program_info(program_path: Path):
    """
    Loads the 'func' and reliably extracts the equation string by reading
    the function definition block directly from the file.
    """
    if not program_path.is_file():
        raise FileNotFoundError(f"Program file not found: {program_path}")
    
    with open(program_path, 'r') as f:
        lines = f.readlines()

    equation_str = "Equation not found"
    try:
        func_lines = []
        in_func = False
        for line in lines:
            if line.strip().startswith('def func('):
                in_func = True
                continue
            if in_func:
                if 'EVOLVE-BLOCK-END' in line:
                    break
                if line.strip() and not line.strip().startswith('#'):
                    func_lines.append(line.strip())

        return_line = ""
        for line in reversed(func_lines):
            if line.startswith('return '):
                return_line = line
                break
        
        if return_line:
            returned_value = return_line.split('return', 1)[1].strip()
            
            if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', returned_value):
                assignment_pattern = re.compile(rf"^\s*{re.escape(returned_value)}\s*=\s*(.*)")
                for line in reversed(func_lines):
                    match = assignment_pattern.match(line)
                    if match:
                        equation_str = match.group(1).strip()
                        break
                else:
                    equation_str = returned_value
            else:
                equation_str = returned_value
        
    except Exception as e:
        print(f"Warning: Could not parse equation from {program_path.name}: {e}", file=sys.stderr)
        equation_str = "Failed to parse"

    spec = importlib.util.spec_from_file_location("model_module", str(program_path))
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    
    if not hasattr(model_module, 'func') or not callable(model_module.func):
        raise AttributeError(f"'func' not found or not callable in {program_path}")
        
    return model_module.func, equation_str

def calculate_3d_metrics(P_pred, P_gt):
    """Calculates all required metrics for 3D point clouds."""
    metrics = {}
    
    mse_per_axis = np.mean((P_pred - P_gt)**2, axis=0)
    metrics['x_mse'] = mse_per_axis[0]
    metrics['y_mse'] = mse_per_axis[1]
    metrics['z_mse'] = mse_per_axis[2]

    metrics['mse'] = np.mean(mse_per_axis)
    
    variance_gt = np.sum(np.var(P_gt, axis=0))
    metrics['nmse'] = metrics['mse'] / variance_gt if variance_gt > 0 else np.inf

    dists = cdist(P_pred, P_gt, 'euclidean')
    min_dist_pred_to_gt = np.min(dists, axis=1)
    min_dist_gt_to_pred = np.min(dists, axis=0)
    
    metrics['chamfer'] = np.mean(min_dist_pred_to_gt**2) + np.mean(min_dist_gt_to_pred**2)
    metrics['hausdorff'] = np.max([np.max(min_dist_pred_to_gt), np.max(min_dist_gt_to_pred)])
    
    return metrics

def optimize_parameters_individual_multi_start(progs, params_uv_train, gt_train_dict, num_attempts=10):
    """
    Finds the optimal parameters for each program (x, y, z) individually by
    minimizing MSE on the training data, using a multi-start strategy.
    """
    optimized_params = {}

    def objective_function(params, model_func, X_matrix, y_true_vector):
        try:
            predictions = model_func(X_matrix, params=params)
            if not np.all(np.isfinite(predictions)):
                return np.inf
            mse = np.mean((predictions.squeeze() - y_true_vector.squeeze())**2)
            return mse
        except Exception:
            return np.inf

    for axis in ['x', 'y', 'z']:
        print(f"--- Optimizing parameters for '{axis}' function ---")
        prog_func = progs[axis]
        y_gt_train = gt_train_dict[axis]
        
        best_mse = float('inf')
        best_params_for_axis = None
        
        for i in range(num_attempts):
            initial_params = np.random.rand(10)
            
            result = minimize(
                objective_function,
                initial_params,
                args=(prog_func, params_uv_train, y_gt_train),
                method='L-BFGS-B'
            )
            
            if result.success and result.fun < best_mse:
                best_mse = result.fun
                best_params_for_axis = result.x
                print(f"  Attempt {i+1}/{num_attempts} successful. New best MSE: {best_mse:.6f}")
            elif not result.success:
                 print(f"  Attempt {i+1}/{num_attempts} did not converge. Reason: {result.message}")

        if best_params_for_axis is None:
            print(f"Warning: Optimization failed for axis '{axis}' after {num_attempts} attempts.", file=sys.stderr)
            return None
        
        optimized_params[axis] = best_params_for_axis

    return optimized_params


def evaluate_parametric_problem(problem_dir: Path):
    start_time = time.time()
    
    try:
        params_uv_train_path = problem_dir / "x/X_train_for_eval.npy"
        x_gt_train_path = problem_dir / "x/y_train_for_eval.npy"
        y_gt_train_path = problem_dir / "y/y_train_for_eval.npy"
        z_gt_train_path = problem_dir / "z/y_train_for_eval.npy"

        required_files = [params_uv_train_path, x_gt_train_path, y_gt_train_path, z_gt_train_path]
        if not all(p.exists() for p in required_files):
             print(f"Skipping {problem_dir.name}: One or more training data files not found.", file=sys.stderr)
             return None

        params_uv_train = np.load(params_uv_train_path)
        x_gt_train = np.load(x_gt_train_path)
        y_gt_train = np.load(y_gt_train_path)
        z_gt_train = np.load(z_gt_train_path)
        
        gt_train_dict = {'x': x_gt_train, 'y': y_gt_train, 'z': z_gt_train}
        num_train_datapoints = params_uv_train.shape
        
    except Exception as e:
        print(f"Failed to load training data for {problem_dir.name}: {e}", file=sys.stderr)
        return None


    log_entry = {
        "equation_id": problem_dir.name,
        "gt_equation": f"[x,y,z] = f(u,v) for {problem_dir.name}",
        "num_datapoints": num_train_datapoints,
        "num_eval_datapoints": 0,
        "eval_results": []
    }

    try:
        prog_x, eq_x_str = load_program_info(problem_dir / "x/openevolve_output/best/best_program.py")
        prog_y, eq_y_str = load_program_info(problem_dir / "y/openevolve_output/best/best_program.py")
        prog_z, eq_z_str = load_program_info(problem_dir / "z/openevolve_output/best/best_program.py")
        
        progs = {'x': prog_x, 'y': prog_y, 'z': prog_z}
        discovered_equations = {"x": eq_x_str, "y": eq_y_str, "z": eq_z_str}

        optimized_params = optimize_parameters_individual_multi_start(progs, params_uv_train, gt_train_dict)
        if optimized_params is None:
             raise RuntimeError("Parameter optimization failed for one or more axes.")

        id_metrics = None
        id_data_path = problem_dir / "test_data_eval.npy"
        if id_data_path.exists():
            eval_data = np.load(id_data_path)
            log_entry["num_eval_datapoints"] = len(eval_data)
            params_uv, P_gt = eval_data[:, :2], eval_data[:, 2:]
            
            P_pred = np.stack([
                prog_x(params_uv, params=optimized_params['x']),
                prog_y(params_uv, params=optimized_params['y']),
                prog_z(params_uv, params=optimized_params['z'])
            ], axis=1)

            if np.all(np.isfinite(P_pred)):
                id_metrics = calculate_3d_metrics(P_pred, P_gt)
        
        ood_metrics = None
        ood_data_path = problem_dir / "ood_test_eval.npy"
        if ood_data_path.exists():
            eval_data_ood = np.load(ood_data_path)
            params_uv_ood, P_gt_ood = eval_data_ood[:, :2], eval_data_ood[:, 2:]
            
            P_pred_ood = np.stack([
                prog_x(params_uv_ood, params=optimized_params['x']),
                prog_y(params_uv_ood, params=optimized_params['y']),
                prog_z(params_uv_ood, params=optimized_params['z'])
            ], axis=1)
            
            if np.all(np.isfinite(P_pred_ood)):
                ood_metrics = calculate_3d_metrics(P_pred_ood, P_gt_ood)

        search_time = time.time() - start_time
        
        eval_result = {
            "search_time": search_time,
            "discovered_equations": discovered_equations,
            "id_metrics": id_metrics,
            "ood_metrics": ood_metrics
        }
        log_entry["eval_results"].append(eval_result)

    except Exception as e:
        print(f"Failed to evaluate {problem_dir.name}: {e}", file=sys.stderr)
        
        eval_result = { 
            "search_time": time.time() - start_time, 
            "discovered_equations": discovered_equations if 'discovered_equations' in locals() else None,
            "id_metrics": None, 
            "ood_metrics": None 
        }
        log_entry["eval_results"].append(eval_result)
            
    return log_entry

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parametric_evaluation_optimized.py <path_to_problems_directory> [output_file]", file=sys.stderr)
        sys.exit(1)

    root_path = Path(sys.argv[1])
    output_file_path = Path("results.jsonl")

    if not root_path.is_dir():
        print(f"Error: Provided path '{root_path}' is not a directory.", file=sys.stderr)
        sys.exit(1)
        
    problem_dirs_to_process = []
    if (root_path / 'x').is_dir() and (root_path / 'y').is_dir():
        problem_dirs_to_process.append(root_path)
    else:
        for problem_dir in root_path.iterdir():
            if problem_dir.is_dir() and (problem_dir / 'x').is_dir() and (problem_dir / 'y').is_dir():
                problem_dirs_to_process.append(problem_dir)

    with open(output_file_path, 'a') as f_out:
        for problem_dir in problem_dirs_to_process:
            print(f"\nProcessing {problem_dir.name}...")
            final_log = evaluate_parametric_problem(problem_dir)
            if final_log:
                f_out.write(json.dumps(final_log, cls=NumpyFloatJSONEncoder) + '\n')
    
    print(f"\nEvaluation complete. Results saved to {output_file_path}")