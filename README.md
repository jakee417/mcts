Minimal implementation of adding Monte Carlo Tree Search to an LLM. See [this blog post](https://jakee417.github.io/posts/monte-carlo-tree-search/) for more details.

## How to use
Copy `mcts.ipynb` and overwrite:
-  `get_candidates_fn` - Computes a list of possible actions and their probabilities of occurring.
-  `get_simulation_fn` - Generate a complete solution (python code, math proof, etc.) from a given state.
-  `get_rewards_fn` - Evaluate a complete solution. Return the raw score and whether MCTS should terminate early.

For your use case. Calling `mcts()` with these arguments will run MCTS. If using the implmentation provided in `open_ai.py`, supply your own API key by setting the environmental variable: `OPENAI_API_KEY`.
