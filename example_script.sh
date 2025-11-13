# Starting VLLM server
CUDA_VISIBLE_DEVICES=5 vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype auto --api-key token-abc123 --port 10005

# SurfaceBench-Explicit Dataset
python eval.py --dataset Complex_Composite_Surfaces --searcher_config configs/lasr_llama31_8b.yaml --local_llm_port 10005
python eval.py --dataset Complex_Composite_Surfaces --searcher_config configs/sga_llama31_8b.yaml --local_llm_port 10005
python eval.py --dataset Complex_Composite_Surfaces --searcher_config configs/llmsr_llama31_8b.yaml --local_llm_port 10005

# SurfaceBench-Implicit Dataset
python eval.py --dataset Algebraic_Manifolds_of_Higher_Degree --searcher_config configs/lasr_llama31_8b.yaml --local_llm_port 10005
python eval.py --dataset Algebraic_Manifolds_of_Higher_Degree --searcher_config configs/sga_llama31_8b.yaml --local_llm_port 10005
python eval.py --dataset Algebraic_Manifolds_of_Higher_Degree --searcher_config configs/llmsr_llama31_8b.yaml --local_llm_port 10005

# SurfaceBench-Parametric Dataset | PySR
python eval.py --dataset Transformed_Coordinate_Surfaces --searcher_config configs/pysr_parametric.yaml