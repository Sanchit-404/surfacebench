import numpy as np
import sympy
from pysr import PySRRegressor
from bench.data_classes import Equation, SearchResult, SEDTask
from bench.searchers.base import BaseSearcher

class PySRSearcher(BaseSearcher):
    """
    A unified searcher for PySR to discover both parametric and explicit/Implicit equations.
    The behavior is controlled by the `mode` parameter.
    - 'parametric': Finds equations for each component (x, y, z) as a function of (u, v).
    - 'explicit': Finds a single equation z = f(x, y).
    """
    def __init__(self, pysr_params: dict, mode: str = 'parametric'):
        if mode not in ['parametric', 'explicit']:
            raise ValueError("Mode must be either 'parametric' or 'explicit'")
        self.mode = mode
        self.pysr_params = dict(pysr_params)

        if "extra_sympy_mappings" not in self.pysr_params:
            self.pysr_params["extra_sympy_mappings"] = {
                "sin": sympy.sin,
                "cos": sympy.cos,
                "tan": sympy.tan,
                "exp": sympy.exp,
                "log": sympy.log, #add more(later)
            }

    def _run_pysr_and_package_result(self, X_train, y_train, input_vars, output_var) -> SearchResult:
        """Helper method to run PySR and format the output."""
        print(f"Training PySR for: {output_var} = f({', '.join(input_vars)})...")
        
        pysr_model = PySRRegressor(**self.pysr_params)
        pysr_model.fit(X_train, y_train, variable_names=input_vars)

        if len(pysr_model.equations) == 0:
            print(f"PySR search did not find any equations for {output_var}.")
            return None
            
        best_eq = pysr_model.get_best()
        sympy_expr = best_eq["equation"]
        discovered_expression = str(sympy_expr)

        lambda_fn = sympy.lambdify(input_vars, sympy_expr, 'numpy')

        arg_string = ', '.join([f"{name}: np.ndarray" for name in input_vars])
        program_format = f"""
def equation({arg_string}) -> np.ndarray:
    \"\"\"Discovered function for {output_var}.

    Args:
        {', '.join(input_vars)} (np.ndarray): Input values.

    Returns:
        np.ndarray: Predicted {output_var} value.
    \"\"\"
    return {discovered_expression}
"""
        all_symbols = input_vars + [output_var]
        symbol_descs = [f"variable {name}" for name in input_vars] + [f"output {output_var}"]
        symbol_properties = ['V'] * len(input_vars) + ['O']

        equation = Equation(
            symbols=all_symbols,
            symbol_descs=symbol_descs,
            symbol_properties=symbol_properties,
            expression=discovered_expression,
            sympy_format=sympy_expr,
            lambda_format=lambda_fn,
            program_format=program_format
        )
        return SearchResult(equation=equation, aux={'component': output_var})

    def discover(self, task: SEDTask) -> list[SearchResult]:
        if self.mode == 'parametric':
            return self._discover_parametric(task)
        else:
            return self._discover_explicit(task)

    def _discover_parametric(self, task: SEDTask) -> list[SearchResult]:
        results = []
        output_symbols = ['x', 'y', 'z']
        input_vars = ["u", "v"]
        
        for output_sym in output_symbols:
            if output_sym not in task.samples:
                print(f"Skipping {output_sym}: no data found.")
                continue

            samples = task.samples[output_sym]
            X_train = samples[:, :2]
            y_train = samples[:, 2] 

            result = self._run_pysr_and_package_result(X_train, y_train, input_vars, output_sym)
            if result:
                results.append(result)
            
        return results

    def _discover_explicit(self, task: SEDTask) -> list[SearchResult]:
        samples = task.samples
        X_train = samples[:, :-1]
        y_train = samples[:, -1]

        num_input_features = X_train.shape[1]
        input_vars = [f"x{i}" for i in range(num_input_features)]
        output_var = getattr(task, 'target_variable_name', 'y')

        result = self._run_pysr_and_package_result(X_train, y_train, input_vars, output_var)
        
        return [result] if result else []