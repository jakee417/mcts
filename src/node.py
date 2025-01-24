from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Protocol


class Node(Protocol):
    """Protocol for a node used in MCTS.

    Attributes:
        prob: Input for the pUCB calculation.
        state: Full generated text sequence up until this node.
        type: Metadata of how node is added.
        value: Max reward obtainable from node.
        visits: Number of visits into this node.
        children: Nodes that descend from this node.
        parent: Where this node comes from.
    """

    prob: float
    value: float
    visits: int
    state: str
    type: str
    children: List[Node]

    def backprop(self, value: float): ...


@dataclass
class MCTSNode:
    """MCTS Node capable of backtracking."""

    prob: float
    state: str
    type: str
    value: float = 0
    visits: int = 0
    children: List[Node] = field(default_factory=lambda: [])
    parent: Optional[Node] = None

    def backprop(self, value: float):
        if value > self.value:
            self.value = value
        if self.parent is not None:
            self.parent.backprop(value)
