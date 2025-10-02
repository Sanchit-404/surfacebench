from datetime import datetime
import os
from pathlib import Path
from bench.searchers.base import BaseSearcher
from bench.data_classes import Equation, SEDTask, SearchResult
from pysr import PySRRegressor

custom_loss = """
function eval_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
        return L(Inf)
    end
    return sum( (1000 .* (prediction .- dataset.y) ) .^ 2) / dataset.n
end
"""

class LasrSearcher(BaseSearcher):
    def __init__(self,
                 name, 
                 api_key, model, model_url, prompts_path, 
                 log_path,
                 temp_dir,
                 num_iterations=25, 
                 num_populations=10,
                 early_stopping_condition=None,
                 llm_weight = 0.005,
                 max_num_samples=2000) -> None:
        super().__init__(name)

        self.num_iterations = num_iterations
        self.early_stopping_condition = early_stopping_condition
        self.num_populations = num_populations
        self.max_num_samples = max_num_samples

        self.llm_options = dict(
            active=True,
            weights=dict(
                llm_mutate=llm_weight,
                llm_crossover=llm_weight,
                llm_gen_random=llm_weight,
            ),
            prompt_evol=True,
            prompt_concepts=True,
            num_pareto_context=3,
            api_key=api_key,
            model=model,
            api_kwargs=dict(
                max_tokens=1024,
                url=model_url,
            ),
            http_kwargs=dict(
                retries=5,
                readtimeout=360,
                retry_delays=30,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "x-use-cache": "false",
                },
            ),
            llm_recorder_dir=log_path,
            idea_threshold=30,
            prompts_dir=prompts_path,
            is_parametric=False,
            var_order=None,
        )
        self.log_path = log_path
        self._temp_dir = temp_dir

    def discover(self, task: SEDTask):
        info = task
        datasets = task.samples[:self.max_num_samples]
        set_llm_options = self.llm_options
        set_llm_options['llm_recorder_dir'] = os.path.join(self.log_path, task.name)

        var_names = task.symbols
        var_desc = task.symbol_descs
        var_desc = [f"{d} ({n})" for d,n in zip(var_desc, var_names)]
        if len(var_desc) > 2:
            input_desc = ", ".join(var_desc[1:-1]) + ", and " + var_desc[-1]
        else:
            input_desc = var_desc[-1]
        desc = f"The expression represents {var_desc[0]}, given data on {input_desc}.\n"
        print(desc)

        set_llm_options["llm_context"] = desc
        set_llm_options['var_order'] = {"x" + str(i): s for i, s in enumerate(info.symbols[1:])}
        temp_dir = self._temp_dir

        # Refer to https://github.com/trishullab/LibraryAugmentedSymbolicRegression.jl/blob/lasr-experiments/experiments/model.py
        model = PySRRegressor(
            niterations=self.num_iterations,
            ncyclesperiteration=550,
            # ncycles_per_iteration=100,
            populations=self.num_populations,
            population_size=33,
            maxsize=30,
            binary_operators=["+", "*", "-", "/", "^"],
            unary_operators=[
                "exp",
                "log",
                "sqrt",
                "sin",
                "cos",
            ],
            loss_function=custom_loss,
            early_stop_condition=f"f(loss, complexity) = (loss < {format(float(self.early_stopping_condition), 'f')})"
            if self.early_stopping_condition
            else None,
            verbosity=1,
            temp_equation_file=True,
            tempdir=temp_dir,
            delete_tempfiles=True,
            # llm_options=set_llm_options,
            weight_randomize=0.1,
            should_simplify=True,
            constraints={
                "sin": 10,
                "cos": 10,
                "exp": 20,
                "log": 20,
                "sqrt": 20,
                "pow": (-1, 20),
            },
            nested_constraints={
                "sin": {"sin": 0, "cos": 0},
                "cos": {"sin": 0, "cos": 0},
                "exp": {"exp": 0, "log": 0},
                "log": {"exp": 0, "log": 0},
                "sqrt": {"sqrt": 0},
            },
            progress=True,
        )

        X, y = datasets[:self.max_num_samples, 1:], datasets[:self.max_num_samples, 0]
        y = y.reshape(-1, 1)

        now = datetime.now()
        now_str = now.strftime("%m-%d-%Y_%H-%M-%S-%f")
        # run_log_file=str(os.path.abspath(os.path.join(self.log_path, "run_logs", f"{task.name}_{now_str}.csv")))
        # run_log_file = Path(run_log_file)
        # run_log_file.parent.mkdir(exist_ok=True, parents=True)
        # print(f"Logging to {run_log_file}")
        model.fit(X, y)

        best_equation = Equation(
            symbols=task.symbols,
            symbol_descs=task.symbol_descs,
            symbol_properties=task.symbol_properties,
            expression=str(model.sympy()),
            sympy_format=model.sympy(),
            lambda_format=model.predict
        )

        lasr_score = None
        for i, row in model.equations_.iterrows():
            if row.equation == best_equation.expression:
                lasr_score = row.score
                break

        return [SearchResult(
            equation=best_equation,
            aux={"lasr_score": lasr_score}
        )]