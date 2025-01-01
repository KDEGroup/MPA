# MPA

Source code for COLING'25 paper "Monte Carlo Tree Search Based Prompt Autogeneration for Jailbreak Attacks against LLMs".

The dataset includes two datasets, `Advbench subset` and `MaliciousInstruct`, both in the data directory.



## Getting started

To get started, install dependencies: `pip install fachat==0.2.23 transformers openai anthropic`

For experiments on GPT models, make sure you have the `OPENAI_API_KEY`.





## Run experiments

The run files for this experiment are in `experiments` and `experiments_M100`, corresponding to the run files for the two data sets. An example of a run command is `bash experiments/exps_llama3_8b.sh`.There is 1 main files:

- `main.py`: runs mcts on all models with logprobs (HuggingFace and GPT models).

 













