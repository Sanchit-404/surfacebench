import os
import sys
import yaml
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser, Namespace
project_root_dir = os.path.dirname(os.path.abspath(__file__))
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)
from bench.datamodules import get_datamodule
from bench.pipelines import EvaluationPipeline
from methods.llmsr.searcher import LLMSRSearcher
from methods.llmsr import config as llmsr_config
from methods.llmsr import sampler as llmsr_sampler
from dotenv import load_dotenv
load_dotenv()

parser = ArgumentParser()
parser.add_argument('--searcher_config', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--ds_root_folder', type=str, default=None)
parser.add_argument('--resume_from', type=str, default=None)
parser.add_argument('--problem_name', type=str, default=None)
parser.add_argument('--local_llm_port', type=int, default=None)
args = parser.parse_args()

now = datetime.now()
now_str = now.strftime("%m-%d-%Y_%H-%M-%S-%f")

dm = get_datamodule(name=args.dataset, hdf5_path=args.ds_root_folder or 'parametric_dataset.h5')
dm.setup()



with open(args.searcher_config) as f:
    searcher_cfg = yaml.safe_load(f)
searcher_cfg = Namespace(**searcher_cfg)

print(f"Loading searcher configuration from: {args.searcher_config}")
with open(args.searcher_config, 'r') as f:
    searcher_cfg = Namespace(**yaml.safe_load(f))

if args.resume_from is None:
    output_path = Path(f"logs/{dm.category_name}/{searcher_cfg.name}/{now_str}")
    output_path.mkdir(parents=True, exist_ok=True)
else:
    output_path = Path(args.resume_from)
searcher_log_path = output_path / "search_logs"
searcher_log_path.mkdir(exist_ok=True, parents=True)

temp_dir = Path("logs/tmp")
temp_dir.mkdir(exist_ok=True, parents=True)

if searcher_cfg.api_type == "hfinf":
    api_key = os.environ['HFINF_API_KEY']
elif searcher_cfg.api_type == "vllm":
    api_key = os.environ['VLLM_API_KEY']
    searcher_cfg.api_url = searcher_cfg.api_url.format(args.local_llm_port)
elif searcher_cfg.api_type == "openai":
    api_key = os.environ['OPENAI_API_KEY']
else:
    api_key = None

if searcher_cfg.class_name == 'LLMSRSearcher':
    sys.path.append(os.path.join(os.path.dirname(__file__), "methods"))
    from methods.llmsr.searcher import LLMSRSearcher
    from methods.llmsr import config, sampler
    # os.environ["LLM_SR_SERVER_PORT"] = str(args.port)
    
    exp_conf = config.ExperienceBufferConfig(
        num_islands=searcher_cfg.num_islands
    )
    cfg = config.Config(
        experience_buffer=exp_conf,
        use_api = searcher_cfg.api_type != 'local', 
        api_model = searcher_cfg.api_model,
        samples_per_prompt = searcher_cfg.samples_per_prompt,
    )
    sampler_class = lambda samples_per_prompt: sampler.LocalLLM(
        samples_per_prompt=samples_per_prompt,
        local_llm_url=searcher_cfg.api_url,
        api_url=searcher_cfg.api_url,
        api_key=api_key,
    )
    searcher = LLMSRSearcher(searcher_cfg.name,
                            cfg, 
                            sampler_class,
                            global_max_sample_num=searcher_cfg.global_max_sample_num, 
                            log_path=searcher_log_path)
elif searcher_cfg.class_name == 'LasrSearcher':
    sys.path.append(os.path.join(os.path.dirname(__file__), "methods"))
    from methods.lasr.searcher import LasrSearcher

    searcher = LasrSearcher(
        name=searcher_cfg.name,
        api_key=api_key,
        model=searcher_cfg.api_model,
        model_url=searcher_cfg.api_url,
        prompts_path='methods/lasr/prompts/',
        log_path=searcher_log_path,
        temp_dir=temp_dir,
        num_iterations=searcher_cfg.num_iterations,
        num_populations=searcher_cfg.num_populations,
        llm_weight=searcher_cfg.llm_weight,
        early_stopping_condition=searcher_cfg.early_stopping_condition,
        max_num_samples=searcher_cfg.max_num_samples,
    )
elif searcher_cfg.class_name == 'SGASearcher':
    sys.path.append(os.path.join(os.path.dirname(__file__), "methods",  "sga_sr"))
    from methods.sga_sr.searcher import SGASearcher
    searcher = SGASearcher(
        name=searcher_cfg.name,
        root=Path("methods/sga_sr").absolute(),
        path=str(searcher_log_path.absolute()),
        python_path=os.environ['SGA_PYTHON_PATH'],
        dataset_name=args.dataset,
        dataset_path=args.ds_root_folder,
        llm_api_url=searcher_cfg.api_url,
        llm_model=searcher_cfg.api_model,
        llm_api_key=api_key,
    )
elif searcher_cfg.class_name == 'PySRSearcher':
    sys.path.append(os.path.join(os.path.dirname(__file__), "methods"))
    from methods.PySR.pysr_searcher import PySRSearcher
    pysr_params = {
        "niterations": searcher_cfg.niterations,
        "binary_operators": searcher_cfg.binary_operators,
        "unary_operators": searcher_cfg.unary_operators,
        "nested_constraints": getattr(searcher_cfg, 'nested_constraints', None),
        "procs": searcher_cfg.procs,
    }
    search_mode = getattr(searcher_cfg, 'mode', 'explicit')

    searcher = PySRSearcher(pysr_params, mode=search_mode)
else:
    raise ValueError

problems = dm.problems
if args.problem_name is not None:
    problems = list(filter(lambda p: p.equation_idx == args.problem_name, problems))
print(f"Total number of problems: {len(problems)}")

pipeline = EvaluationPipeline()
pipeline.evaluate_problems(problems, 
                           searcher, 
                           output_path)