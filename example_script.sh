# Starting VLLM server
CUDA_VISIBLE_DEVICES=5 vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype auto --api-key token-abc123 --port 10005

# LSR-Transform Dataset
python eval.py --dataset lsrtransform --searcher_config configs/llmdirect_llama31_8b.yaml --local_llm_port 10005
python eval.py --dataset lsrtransform --searcher_config configs/lasr_llama31_8b.yaml --local_llm_port 10005
python eval.py --dataset lsrtransform --searcher_config configs/sga_llama31_8b.yaml --local_llm_port 10005
python eval.py --dataset lsrtransform --searcher_config configs/llmsr_llama31_8b.yaml --local_llm_port 10005

# Bio-Pop-Growth Dataset
python eval.py --dataset bio_pop_growth --searcher_config configs/llmdirect_llama31_8b.yaml --local_llm_port 10005
python eval.py --dataset bio_pop_growth --searcher_config configs/lasr_llama31_8b.yaml --local_llm_port 10005
python eval.py --dataset bio_pop_growth --searcher_config configs/sga_llama31_8b.yaml --local_llm_port 10005
python eval.py --dataset bio_pop_growth --searcher_config configs/llmsr_llama31_8b.yaml --local_llm_port 10005

# Chem React Kinetics
python eval.py --dataset chem_react --searcher_config configs/llmdirect_llama31_8b.yaml --local_llm_port 10005
python eval.py --dataset chem_react --searcher_config configs/lasr_llama31_8b.yaml --local_llm_port 10005
python eval.py --dataset chem_react --searcher_config configs/sga_llama31_8b.yaml --local_llm_port 10005
python eval.py --dataset chem_react --searcher_config configs/llmsr_llama31_8b.yaml --local_llm_port 10005

# Matsci SS
python eval.py --dataset matsci --searcher_config configs/llmdirect_llama31_8b.yaml --local_llm_port 10005
python eval.py --dataset matsci --searcher_config configs/lasr_llama31_8b.yaml --local_llm_port 10005
python eval.py --dataset matsci --searcher_config configs/sga_llama31_8b.yaml --local_llm_port 10005
python eval.py --dataset matsci --searcher_config configs/llmsr_llama31_8b.yaml --local_llm_port 10005

# Phys oscillator
python eval.py --dataset phys_osc --searcher_config configs/llmdirect_llama31_8b.yaml --local_llm_port 10005
python eval.py --dataset phys_osc --searcher_config configs/lasr_llama31_8b.yaml --local_llm_port 10005
python eval.py --dataset phys_osc --searcher_config configs/sga_llama31_8b.yaml --local_llm_port 10005
python eval.py --dataset phys_osc --searcher_config configs/llmsr_llama31_8b.yaml --local_llm_port 10005