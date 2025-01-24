from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from src.node import MCTSNode, Node
import logging

logger = logging.getLogger(__name__)

C = 1.0
C_BASE = 1.0


def compute_beta(node: Node, c: float, c_base: float) -> float:
    """Computes the beta coefficient for a parent node."""
    return np.log((node.visits + c_base + 1) / c_base) + c


def p_ucb_value(node: Node, parent_node: Node, beta: float) -> float:
    """Computes the p ucb value for a given node."""
    return node.value + beta * node.prob * np.sqrt(np.log(parent_node.visits)) / (
        1 + node.visits
    )


def p_ucb_select(node: Node, c: float, c_base: float) -> Optional[Node]:
    """Finds the child node with the highest p-UCB value."""
    beta = compute_beta(node=node, c=c, c_base=c_base)
    max_p_ucb = -np.inf
    max_node = None
    for child in node.children:
        p_ucb = p_ucb_value(
            node=child,
            parent_node=node,
            beta=beta,
        )
        if p_ucb > max_p_ucb:
            max_node = child
            max_p_ucb = p_ucb
    return max_node


def get_next_actions(
    node: Node,
    get_candidates_fn: Callable[[Node], Tuple[List[str], List[float]]],
) -> List[Node]:
    """Get next actions based on the current node."""
    candidates, probabilities = get_candidates_fn(node)
    return [
        MCTSNode(
            prob=prob,
            state=(node.state + " " + c).strip(),
            parent=node,
            type="expansion",
        )
        for c, prob in zip(candidates, probabilities, strict=True)
    ]


def _mcts_single_step(
    root: Node,
    get_candidates_fn: Callable[[Node], Tuple[List[str], List[float]]],
    get_simulation_fn: Callable[[Node], str],
    get_rewards_fn: Callable[[Node, str], Tuple[float, bool]],
    c: float,
    c_base: float,
) -> Tuple[Node, Optional[Node], List[Dict[str, Any]], float, bool]:
    step: List[Dict[str, Any]] = []
    node = root
    node.visits += 1

    # Selection
    while (selected_node := p_ucb_select(node=node, c=c, c_base=c_base)) is not None:
        node = selected_node
        node.visits += 1
        step.append({"selection": node})

    # Expansion
    node.children = get_next_actions(
        node=node,
        get_candidates_fn=get_candidates_fn,
    )
    step.append({"expansion": node.children})

    # Simulation
    simulation = get_simulation_fn(node)
    step.append({"simulation": simulation})

    # Evaluation
    reward, early_stop = get_rewards_fn(node, simulation)
    step.append({"reward": (node.state + " " + simulation, reward)})

    # backprop reward up the tree
    node.backprop(reward)

    # early termination if we found a perfect reward
    if early_stop:
        node.children.append(
            MCTSNode(
                prob=1.0,
                state=node.state + " " + simulation,
                parent=node,
                type="simulation",
            )
        )
        node = node.children[-1]
        step.append({"early_termination": node})

    return root, node, step, reward, early_stop


def mcts(
    get_candidates_fn: Callable[[Node], Tuple[List[str], List[float]]],
    get_simulation_fn: Callable[[Node], str],
    get_rewards_fn: Callable[[Node, str], Tuple[float, bool]],
    max_rollouts: int = 8,
    c: float = C,
    c_base: float = C_BASE,
    verbose: bool = True,
) -> Tuple[Node, Optional[Node], List[Dict[str, Any]]]:
    """Run MCTS for a given prompt.

    Args:
        get_candidates_fn: Compute a list of tokens and probabilities for a given state.
        get_simulation_fn: Generate a simulated response for a given state.
        get_rewards_fn: Evaluate a solution for a given state & simulation.
            Returns the raw score and whether a correct solution has been found.
        max_rollouts: Number of MCTS evaluations to run.
        c: exploration coefficient.
        c_base: exploration coefficient.
        verbose: whether to log step level information.

    Returns:
        Root and last evaluated node along with action history.
    """
    root = MCTSNode(prob=1, state="", type="root")
    node: Optional[Node] = None
    history: List[Dict[str, Any]] = [{"root": root}]
    logger.info({"actions": [list(i.keys()) for i in history]})
    for i in range(max_rollouts):
        # Take a mcts step always starting from the root.
        root, node, step, reward, early_stop = _mcts_single_step(
            root=root,
            get_candidates_fn=get_candidates_fn,
            get_simulation_fn=get_simulation_fn,
            get_rewards_fn=get_rewards_fn,
            c=c,
            c_base=c_base,
        )

        # Extend the history with the current step and print.
        history.extend(step)
        if verbose:
            logger.info(
                {
                    "step": i,
                    "actions": [list(i.keys()) for i in step],
                    "reward": round(reward, 3),
                }
            )

        # If we hit our early stop condition, stop.
        if early_stop:
            break

    return root, node, history
