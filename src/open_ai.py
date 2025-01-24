from functools import cache
import json
import os
from typing import Callable, List, Optional, Tuple
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from openai.types.chat.chat_completion import ChatCompletion

from src.node import Node

load_dotenv()

CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@cache
def get_completion(
    system_prompt: str,
    translation: str,
    max_completion_tokens: Optional[int],
    top_logprobs: int,
) -> ChatCompletion:
    return CLIENT.chat.completions.create(
        model="gpt-4o-mini",
        seed=1,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": translation,
            },
        ],
        logprobs=True,
        top_logprobs=top_logprobs,
        max_completion_tokens=max_completion_tokens,
    )


def get_candidates_fn(
    prompt: str,
    max_completion_tokens: int,
    top_logprobs: int,
    minimum_candidate_token_length: int = 0,
) -> Callable[[Node], Tuple[List[str], List[float]]]:
    def _get_candidates_fn(node: Node) -> Tuple[List[str], List[float]]:
        """Get the next candidates given a state.

        Notes:
            Since top_logprobs returns the highest k probability tokens given the
            latest sequence, we can extract n * k total choices across all the completion tokens.
            Although we should consider the total probability:

            p(tokens) = p(token_n | token_{n - 1}) * p(token_{n - 1} | token_{n - 2}) * ...

            but instead use p(token_n | token_{n - 1}) to not downweight longer sequences.
        """
        if minimum_candidate_token_length > max_completion_tokens:
            raise ValueError(
                f"max_completion_tokens must be > than minimum_candidate_token_length. Found {max_completion_tokens} and {minimum_candidate_token_length}."
            )
        completion = get_completion(
            system_prompt=prompt,
            translation=node.state,
            max_completion_tokens=max_completion_tokens,
            top_logprobs=top_logprobs,
        )
        if len(completion.choices) > 0:
            choice = completion.choices[0]
            if (
                (logprobs := choice.logprobs) is not None
                and (contents := logprobs.content) is not None
                and len(contents) > 0
            ):
                candidates: List[str] = []
                probs: List[float] = []
                for i in range(minimum_candidate_token_length, max_completion_tokens):
                    prefix = "".join([i.token for i in contents[:i]])
                    content = contents[i]
                    candidates += [prefix + i.token for i in content.top_logprobs]
                    probs += [float(np.exp(i.logprob)) for i in content.top_logprobs]
                return (candidates, probs)
        return ([], [])

    return _get_candidates_fn


def get_simulation_fn(prompt: str) -> Callable[[Node], str]:
    def _get_simulation_fn(node: Node) -> str:
        """Get a simulation starting from a node."""
        completion = get_completion(
            system_prompt=prompt,
            translation=node.state,
            max_completion_tokens=None,
            top_logprobs=1,
        )
        if len(completion.choices) > 0:
            choice = completion.choices[0]
            if (text := choice.message.content) is not None:
                return text
        return ""

    return _get_simulation_fn


def get_reward_fn(prompt: str) -> Callable[[Node, str], Tuple[float, bool]]:
    def _get_reward_fn(node: Node, simulation: str) -> Tuple[float, bool]:
        """Get a reward of a node as an average of json values."""
        completion = get_completion(
            system_prompt=prompt,
            translation=node.state + " " + simulation,
            max_completion_tokens=None,
            top_logprobs=1,
        )
        if len(completion.choices) > 0:
            choice = completion.choices[0]
            if (content := choice.message.content) is not None:
                try:
                    score_dict = json.loads(content)
                    score = float(np.mean(list(score_dict.values())))
                    return score, score == 1.0
                except Exception as e:
                    print(e)
        return 0.0, False

    return _get_reward_fn
