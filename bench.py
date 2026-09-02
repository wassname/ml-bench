"""wassname-ml-bench: grade a model's answer to a research design question against a rubric.

One item per markdown file in `items/`. The candidate sees only `prompt` and answers at minimal
reasoning effort. A judge sees the answer, the reference answer, and the rubric, and marks each
rubric point hit or missed with cited answer lines. Score is weighted hits minus weighted traps.

    uv run bench.py --smoke                       # offline self-test, no API key
    uv run bench.py --dry-run --model M           # print the call count, spend nothing
    uv run bench.py --model openrouter/x-ai/grok-4.1-fast
    uv run inspect view --log-dir logs
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import math
import os
import re
import statistics
from datetime import date
from functools import cache
from pathlib import Path
from typing import Literal

import httpx
import yaml
from inspect_ai import Epochs, Task, task
from markdown_it import MarkdownIt
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.hooks import Hooks, RunEnd, SampleEnd, TaskEnd, TaskStart, hooks
from inspect_ai.model import (
    CachePolicy,
    ChatMessageUser,
    GenerateConfig,
    Model,
    ModelOutput,
    ResponseSchema,
    get_model,
)
from inspect_ai.scorer import Score, Target, accuracy, mean, scorer, stderr
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import json_schema
from openai import APIError, ContentFilterFinishReasonError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

ITEMS = Path(__file__).parent / "items"
# Bump when items, rubric, or the answer budget change: it busts the cache and drops older logs
# out of the comparison, so a table never mixes two versions of the same question.
# Still v94: no item, rubric or answer budget differs from the single-judge sweep, so the panel
# re-grades the answers already in the cache and pays for judging only. The judge is recorded per
# log and the report keys on it, so the panel table and the single-judge table stay apart.
# v96, not v95: v95 exists in the logs from a question set that was rolled back, and reusing the
# name would let those answers into this table. The questions are unchanged from v94; what changed
# is who serves the answer, which changes the answer, so the two must not share a table. -- CLAUDE
# v97: the questions are unchanged, but every answer is drawn differently. Two changes, both in
# the request. The effort arm is each model's lowest listed rung instead of a global "minimal" that
# 28 of the 32 models do not have and silently replaced with their own default. And the answer
# budget is 4000 tokens with a continuation turn, equal for every model, instead of 40,000.
RUBRIC_VERSION = "v97"
# Stamped on every chart, so an image that travels without the page keeps its source.
SITE_URL = "wassname.github.io/ml-bench"
# A panel, one company each, because judge choice moves a score more than sampling does. Measured
# on v93 over 219 model-item cells graded twice (scripts/scratch/judge_variance.py): three passes
# of one judge disagree by sd 0.035 to 0.071, which averages down, while two judges disagree by
# 0.104 on average and 0.93 at worst, which does not average down with one judge. The gaps between
# the top models are smaller than that, so one judge cannot rank them.
# google/gemini-3.6-flash was the first pick and failed selection twice over: it 400s with
# "Reasoning is mandatory for this endpoint and cannot be disabled" on the longer judge prompts,
# 7 of 12 items, and under a strict schema it fences its JSON, which is what got the mistral judge
# dropped in 611602a. deepseek-v4-flash-0731 calibrates well but shares a company with the seat
# above it, which would leave a DeepSeek candidate one judge short of everyone else.
# claude-sonnet-5 held the third seat and came out on price: $12.90 of a $14.70 panel sweep, for an
# anchor gap of 0.989 against glm-5.2's 1.008 at $2.70.
# Five seats, wassname's own list, one company each and cheap enough to re-run. Four of the five
# are open weights, which is the point: the answers are frozen in the logs, so grading is the only
# step anyone would ever repeat, and an open judge cannot change under its own name.
JUDGE_PANEL = (
    "openrouter/openai/gpt-oss-120b",
    "openrouter/qwen/qwen3.7-flash",
    "openrouter/deepseek/deepseek-v4-flash-0731",
    "openrouter/thinkingmachines/inkling-small",
    "openrouter/google/gemma-4-31b-it",
)
# The panel identity, stored in each log so a table never mixes two panels.
JUDGE_MODEL = "+".join(sorted(m.removeprefix("openrouter/") for m in JUDGE_PANEL))
# One sample per panel member. The budget buys judges rather than passes because judge choice
# moves a score 0.104 and a repeat sample moves it 0.035 to 0.071.
JUDGE_PASSES = 1
# (off-topic, gold) per judge, from `just calibrate`. Fixed artifacts, not the model pool, so these
# constants do not move when the roster does. Measured at v94 and still in use: `items/` was frozen
# before the v94 bump, so they are anchors on the same questions. Re-measure when an item changes.
# A judge needs gold minus off-topic above 0.5 to sit on the panel: below that it cannot tell the
# reference answer from an unrelated one, and dividing by the gap would amplify its noise.
JUDGE_ANCHORS: dict[str, tuple[float, float]] = {
    # scripts/scratch/read_anchors.py over logs/calibration, pooled over every v94 calibration run,
    # so a re-measurement adds samples rather than replacing them.
    # The five seats.
    "openai/gpt-oss-120b": (-0.0158, +0.9962),  # gap 1.012, 36 gold and 36 off-topic
    "qwen/qwen3.7-flash": (-0.0037, +0.9923),  # gap 0.996, 24 gold and 24 off-topic
    "deepseek/deepseek-v4-flash-0731": (-0.0076, +0.9970),  # gap 1.005, 24 gold and 24 off-topic
    "thinkingmachines/inkling-small": (-0.0200, +0.9877),  # gap 1.008, 24 gold and 24 off-topic
    "google/gemma-4-31b-it": (+0.0103, +0.9901),  # gap 0.980, 24 gold and 24 off-topic
    # Measured but not seated. Every one clears the 0.5 bar, so the choice is company, price and
    # whether the weights are downloadable, not validity.
    "google/gemini-3.1-flash-lite": (+0.0008, +0.9198),  # gap 0.919, the fallback Google seat
    "google/gemma-4-26b-a4b-it": (+0.0291, +0.9474),  # gap 0.918
    "meta/muse-glimmer-30b": (-0.0076, +0.9967),  # gap 1.004
    "deepseek/deepseek-v4-pro-0813": (+0.0076, +0.9889),  # gap 0.981, the v94 single judge
    "openai/gpt-5.6-luna": (-0.0076, +0.9963),  # gap 1.004
    "z-ai/glm-5.2": (-0.0076, +1.0000),  # gap 1.008
    "minimax/minimax-m3": (-0.0076, +0.9408),  # gap 0.948
    "qwen/qwen3.5-397b-a17b": (+0.0014, +0.9708),  # gap 0.969
    "anthropic/claude-sonnet-5": (-0.0152, +0.9735),  # gap 0.989, off the panel on price
}
# RE-Bench normalization (https://metr.org/blog/2024-11-22-evaluating-r-d-capabilities-of-llms/):
# 0 = the obvious starting answer the question rejects, 1 = the reference answer, and above 1 is
# reachable by beating the reference. The rubric points cover [0, 1]; `beyond_reference` is the
# only headroom above it, worth at most this much.
BEYOND_ID = "beyond_reference"
BEYOND_WEIGHT = 0.5
JUDGE_TEMPERATURE = 0.7
# A thinking budget, equal for every model, so the table is not partly measuring how long each lab
# lets its model think. Measured 2026-08-22 over 572 logged answers: prose is near constant across
# models (median 997 tokens) and the 30x spread in output is all thinking, so this bounds thinking
# and `FINAL_ANSWER_MAX_TOKENS` leaves the answer its own room
# (scripts/scratch/answer_vs_thinking.py). Nothing is forced to spend it.
# Not every provider honours it as a thinking cap. deepseek stops dead at it, x-ai does not count
# reasoning against it at all: grok-4.6 logged out=13,674 reasoning=13,193 stop_reason=stop under
# this value (docs/audits/job_3.md). So it is a requested budget, and a cross-provider token
# comparison has to read the logged reasoning tokens, not this number.
ANSWER_MAX_TOKENS = 4_000
# The continuation turn writes prose, not thinking, so it needs room for the longest answer the
# bench has seen. Measured over 572 logged answers: prose alone is median 997 tokens, p99 4436,
# max 5795 (scripts/scratch/answer_vs_thinking.py). A cap under that would truncate the answer
# turn itself and score the model 0 for the harness's reason, not its own.
FINAL_ANSWER_MAX_TOKENS = 6000
# Models whose OpenRouter record does not describe what their provider accepts, so the rung is
# measured instead of read. Applies to both turns.
#
# o1: the record lists no efforts, then the provider rejects the `none` that `{"enabled": false}`
# becomes ("Unsupported value: 'none' ... Supported values are: 'low', 'medium', and 'high'"). An
# empty dict runs it at its own default, which is the best available.
#
# minimax-m3: the record lists no efforts either, but `minimal` is real and is the only setting that
# ever produces prose. With no field, `{"enabled": false}`, or `{"max_tokens": 1000}` it thinks past
# every cap and returns an empty answer, which cost it 6 of 12 items in the first v97 sweep. At
# `minimal` it writes 3176 characters (scripts/scratch/filter_probe.py, 2026-08-22).
EFFORT_OVERRIDE = {"openai/o1": {}, "minimax/minimax-m3": {"reasoning": {"effort": "minimal"}}}
# Thinking rungs, quietest first. No model lists all of them and the names are not comparable
# across labs, so the arm is "this model's lowest rung", resolved per model by `_lowest_effort`,
# rather than one level named here. `none` is thinking switched off, not a low level of it, so it
# is excluded from the candidate arm and used only on the forced-answer turn.
EFFORT_RUNGS = ("minimal", "low", "medium", "high", "xhigh", "max")
# The sentinel for that per-model resolution. Any other value is an explicit override and files as
# its own variant row.
REASONING = "lowest"
# What to call the default arm in the report. It is not one level: the rung differs per model, and
# each log records the rung it actually got in `effort`.
EFFORT_ARM = "lowest listed"
# Who is allowed to serve an answer. Unset, OpenRouter picks by load and price, and
# deepseek-v4-flash-0731 alone has 28 endpoints of which 7 are fp4 and 6 do not say. An fp4
# serving of a model is not the model, so a sweep without this is partly measuring the roulette.
# require_parameters keeps out a provider that would silently drop the reasoning effort we send,
# which is how a model told to think less ends up thinking 12k tokens a question. -- CLAUDE
# fp4 is a different model wearing the same name, so it is out. The filter cannot be sent blindly:
# every closed model reports `unknown`, first-party included, and the allow-list then matches no
# endpoint and the whole request 404s. So it is sent only to models that have a real quantization
# to filter on. -- CLAUDE
GOOD_QUANTS = ["fp8", "int8", "bf16", "fp16"]
# Known hosts first. None of them is perfect, all of them beat the long tail of unknowns that
# OpenRouter would otherwise pick by price. First-party is added per model, ahead of these.
PREFERRED_HOSTS = ["Groq", "DeepInfra", "Fireworks", "Together", "BaseTen"]
# max_tokens includes thinking and the required JSON, which runs about 1k-2k tokens. -- CODEX
JUDGE_MAX_TOKENS = 4096
# Per seat, and a seat is a whole provider to itself.
JUDGE_MAX_CONNECTIONS = 16
MAX_CONNECTIONS = 8
MAX_RETRIES = 6
# The roster. `dev` is for the fix-and-rerun loop, with gemma-3-4b-it as the weak model a valid item
# must rank last. `hle` is the score-vs-cost Pareto front of Humanity's Last Exam (Artificial
# Analysis), which is the comparison this bench exists to disagree with.
MODELS = {
    "deepseek": [
        "openrouter/deepseek/deepseek-v4-flash",
        "openrouter/deepseek/deepseek-v4-flash-0731",
        "openrouter/deepseek/deepseek-v4-pro-0813",
    ],
    "dev": [
        "openrouter/google/gemma-3-4b-it",
        "openrouter/google/gemma-4-31b-it",
        "openrouter/openai/gpt-5.6-luna",
        "openrouter/deepseek/deepseek-v4-flash-0731",
    ],
    # One family at five sizes, so the chart has a cheap left anchor and a scaling curve inside a
    # single model family. Extrapolating the Pareto line to (0,0) would invent a point that cannot
    # even be drawn on a log cost axis; a real 9B can.
    "qwen-sizes": [
        "openrouter/qwen/qwen3.5-9b",
        "openrouter/qwen/qwen3.5-27b",
        "openrouter/qwen/qwen3.5-35b-a3b",
        "openrouter/qwen/qwen3.5-122b-a10b",
        "openrouter/qwen/qwen3.5-397b-a17b",
    ],
    "hle": [
        "openrouter/qwen/qwen3.5-9b",
        "openrouter/openai/gpt-oss-120b",
        # Two 2023-2024 flagships, each the frontier of its own month, so the release-date axis
        # starts before this table's cheap models. Neither takes a reasoning effort setting.
        "openrouter/openai/gpt-4",
        "openrouter/openai/o1",
        # Six older flagships, so the release-date axis is not cheap models on the left and
        # frontier ones on the right.
        "openrouter/qwen/qwen3.5-27b",
        "openrouter/qwen/qwen3.6-27b",
        "openrouter/z-ai/glm-5.1",
        "openrouter/x-ai/grok-4.5",
        "openrouter/openai/gpt-5.5",
        "openrouter/anthropic/claude-opus-4.8",
        "openrouter/deepseek/deepseek-v4-flash-0731",
        "openrouter/minimax/minimax-m3",
        "openrouter/xiaomi/mimo-v2.5-pro",
        "openrouter/deepseek/deepseek-v4-pro-0813",
        "openrouter/z-ai/glm-5.2",
        "openrouter/z-ai/glm-5.3",
        "openrouter/meta/muse-spark-1.2",
        "openrouter/google/gemini-3.6-flash",
        "openrouter/google/gemini-3.7-flash",
        "openrouter/x-ai/grok-4.6",
        "openrouter/qwen/qwen3.8-max",
        "openrouter/moonshotai/kimi-k3",
        "openrouter/openai/gpt-5.6-luna",
        "openrouter/openai/gpt-5.6-terra",
        "openrouter/openai/gpt-5.6-sol",
        "openrouter/anthropic/claude-opus-5",
        "openrouter/anthropic/claude-sonnet-5",
        "openrouter/anthropic/claude-fable-5",
        "openrouter/anthropic/claude-haiku-4.5",
        "openrouter/thinkingmachines/inkling",
        # An unnamed lab's model behind OpenRouter's stealth slug, free while it is cloaked, and a
        # DeepSeek experimental. Neither is on Artificial Analysis, so both sit out the index fits.
        "openrouter/stealth/ox-alpha",
        "openrouter/deepseek/deepseek-v4-flash-vision-exp",
    ],
}
APP_HEADERS = {
    "HTTP-Referer": "https://github.com/wassname/wassname-ml-bench",
    "X-OpenRouter-Title": "wassname-ml-bench",
}


def _openrouter_model(name: str) -> Model:
    return get_model(
        name,
        stream=True,
        # read is per chunk, not per request, so a model that thinks for an hour is fine and only
        # a stream that stops sending dies. With read=None three fill lanes hung for over two
        # hours on ESTAB sockets that never delivered another byte.
        http_client=httpx.AsyncClient(timeout=httpx.Timeout(None, connect=5.0, read=300.0)),
    )


async def _retry_stream_error(call, attempts: int = 6):
    """Retry a mid-stream OpenRouter failure.

    OpenRouter answers 200, opens the SSE stream, then emits an error event, so the openai SDK
    raises a bare `APIError` after the response started. inspect's `max_retries` does not cover
    that, and one raised sample errors the whole log, which drops the model from the table.

    Six attempts back off 1+2+4+8+16 = 31s. Four spanned only 7s, and that lost two samples each
    on gpt-5.6-luna and gpt-5.6-sol to one provider blip.
    """
    for attempt in range(attempts):
        try:
            return await call()
        except APIError as error:
            if attempt == attempts - 1:
                raise
            print(f"openrouter stream error, retry {attempt + 1}/{attempts - 1}: {error}")
            await asyncio.sleep(2 ** attempt)


async def _refusal_reason(model: str, prompt: str) -> str:
    """The provider's own words for a refusal, which the openai SDK drops.

    `ContentFilterFinishReasonError.__init__` takes no arguments, so the exception carries no
    payload at all: `LengthFinishReasonError` keeps its completion snapshot and this one does not.
    The raw OpenRouter body has `native_finish_reason` and `message.refusal` with the real text.
    claude-fable-5 on SV#8 returns "This request triggered restrictions on violative cyber
    content and was blocked under Anthropic's Usage Policy", on a covariance loss question.
    One extra call, only after the retries are spent, so only a persistent refusal pays for it.
    """
    body = {"model": model.replace("openrouter/", ""),
            "messages": [{"role": "user", "content": prompt}], "max_tokens": 200}
    async with httpx.AsyncClient(timeout=300.0) as client:
        reply = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}, json=body)
    choice = reply.json().get("choices", [{}])[0]
    native = choice.get("native_finish_reason")
    refusal = (choice.get("message") or {}).get("refusal")
    return f"{native}: {refusal}" if refusal else f"no refusal text, native_finish_reason={native}"


def _rejoin(partial: str, continuation: str) -> str:
    """Drop a code fence the continuation reopens inside a block that is already open.

    Cut mid-pseudocode, a model starts its continuation with ```python again, and the naive join
    reads "# Forward pass to get h_T and h_```python", which an auditor correctly called a
    corrupted artifact.
    """
    if partial.count("```") % 2 == 0:
        return continuation
    return re.sub(r"^\s*```[a-z]*\n", "", continuation)


async def _finish_answer(state: TaskState, generate: Generate, quiet: dict) -> TaskState:
    """Thinking ate the budget, so hand the work back and ask for the answer alone.

    Without this a model scores 0 on a question it was solving. claude-opus-5 hit a 4000-token cap
    on one item and scored 0.000 where its uncapped answer scored 0.976, and that single cell moved
    its run mean by -0.09 (scripts/scratch/reuse_check.py, 2026-08-21).

    Thinking is switched off here, not lowered, because this turn is for prose. At the model's low
    rung instead, deepseek-v4-flash-0731 spent 4999 of 5000 tokens thinking and wrote one token of
    answer; switched off it wrote 3745 (scripts/scratch/two_turn_probe.py, 2026-08-22).
    """
    partial = state.output.completion
    state.messages.append(ChatMessageUser(
        content="You hit the length limit mid-answer. Continue from exactly where you stopped, "
                "no repetition and no preamble." if partial.strip() else
                "Thinking budget spent. Write the answer now, no more analysis."))
    state = await generate(state, max_tokens=FINAL_ANSWER_MAX_TOKENS, extra_body=quiet)
    # Grade the whole answer. Judging the continuation alone scores the model on a fragment.
    state.output.completion = partial + _rejoin(partial, state.output.completion)
    return state


@solver
def generate_tolerating_content_filter(quiet: dict | None = None,
                                       attempts: int = 3) -> Solver:
    """generate(), except a content_filter rejection is retried and then recorded as unanswered.

    Anthropic models refuse some of these questions, and the refusal is mostly stochastic: over two
    claude-fable-5 runs, TS#10 refused once then answered, OL#6 and AP#1 answered once then
    refused, and only SV#8 refused every time. So retry
    first. The openai SDK raises ContentFilterFinishReasonError rather than returning a stop_reason,
    which errors the whole sample and used to drop the model from the table entirely.

    An answer cut off by `max_tokens` gets one continuation turn, so the budget bounds thinking
    rather than deciding the score.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        for attempt in range(attempts):
            try:
                state = await _retry_stream_error(lambda: generate(state))
                if state.output.stop_reason == "max_tokens":
                    state = await _finish_answer(state, generate, quiet or {})
                return state
            except ContentFilterFinishReasonError:
                print(f"content_filter on {state.sample_id}, attempt {attempt + 1}/{attempts}")
        reason = await _refusal_reason(str(state.model), state.input_text)
        state.output = ModelOutput.from_content(
            "", "", stop_reason="content_filter",
            error=f"refused {attempts}/{attempts} attempts. {reason}",
        )
        return state

    return solve


class PointCheck(BaseModel):
    """One rubric point. The point's own text rides on the field that holds this, not in the
    prompt, so the rule cannot drift from the thing it grades (the Petri pattern in
    skill:ml-debug refs/llm_judges.md)."""

    model_config = ConfigDict(extra="forbid", strict=True)

    lines: str = Field(description="Line numbers of the ANSWER that decide this point, as '34', "
                                   "'34-36', or '12-14,34-36' when the point requires two claims. "
                                   "At most two ranges. Empty when absent.")
    rung: str = Field(pattern=r"^(?:$|(?:0(?:\.0)?|0\.5|1(?:\.0)?)\b)",
                      description="The rung this span meets, copied from the point below. Start "
                                  "with its exact numeric value, then one clause naming it. It may "
                                  "be empty only when the point is absent; empty means 0.0.")
    score: Literal[0.0, 0.5, 1.0] = Field(
        description="Exactly 0.0, 0.5, or 1.0 for how much of THIS point the answer makes. "
        "Never use the point weight as this score. The rungs are in this field's own description. "
        "Judge this point alone.",
    )

class BinaryPointCheck(PointCheck):
    """A point whose rubric explicitly permits only a hit or miss."""

    score: Literal[0.0, 1.0] = Field(
        description="Exactly 0.0 or 1.0. This point explicitly has no partial-credit rung.")


def _rung_score(rung: str) -> float:
    if not rung:
        return 0.0
    return float(re.match(r"^(0\.5|0(?:\.0)?|1(?:\.0)?)", rung).group(1))


def _score_matches_rung(check: PointCheck) -> bool:
    return check.score == _rung_score(check.rung)


def _consistent_score(check: PointCheck) -> float:
    return check.score if _score_matches_rung(check) else 0.0


def grade_model(rubric: list[dict], traps: list[dict]) -> type[BaseModel]:
    """A schema with one required field per rubric point, carrying that point's text.

    This stops two faults by construction rather than by instruction: the judge cannot omit an id,
    and cannot read a rule that has drifted from its point. inspect strips ge/le for
    OpenAI-compatible providers, so `description` is the only channel that reaches the model.

    It does NOT stop one cited span serving two ids, so `_shared_quotes` reports that. A point and a
    trap on one span is the worst case, not a benign one: on lucid_review the same sentence earned
    `cheapest_check` 1.00 and fired `open_the_gate` 0.67, and on TR#9 one span both earned
    `recursion_in_bottleneck` and fired `loop_whole_model`.
    """
    fields: dict = {
        "evidence": (str, Field(description="What the answer proposes, in a sentence or two. "
                                            "Write this first.")),
    }
    for point in rubric:
        check = (BinaryPointCheck if "This point is 1.0 or 0.0." in point["point"]
                 else PointCheck)
        fields[point["id"]] = (check, Field(description=" ".join(point["point"].split())))
    for trap in traps:
        check = (BinaryPointCheck if "This point is 1.0 or 0.0." in trap["point"]
                 else PointCheck)
        fields[trap["id"]] = (check, Field(
            description="TRAP, scoring above 0.0 here loses marks: "
                        + " ".join(trap["point"].split())))
    fields[BEYOND_ID] = (PointCheck, Field(description=(
        "ABOVE THE REFERENCE, the only field that can push this answer past 1.0. Default 0.0. "
        "Score above 0.0 only if the answer does something the reference answer does NOT, that "
        "a researcher would prefer: it fixes a defect in the reference, or it handles a failure "
        "case the reference misses. In `rung`, name the reference's specific defect or omission; "
        "if you cannot name it, this is 0.0. 0.5 for one such improvement. 1.0 for an improvement "
        "that changes which method a researcher would run. More length, more caveats, more detail "
        "on what the reference already says, or a list of alternatives, are all 0.0.")))
    # The judge's half of the exit interview. Never scored, so it costs the judge nothing to
    # complain, and a complaint two judges repeat is evidence about the item.
    fields["judge_note"] = (str, Field(
        description="Friction. Anything that made this hard to grade: a point you could not tell "
                    "apart from its neighbour, a rung that does not fit any answer, a question "
                    "that is ambiguous, an answer right in a way the rubric misses, or a rule you "
                    "could not follow. Say what you would change. Empty when it graded cleanly. "
                    "Never scored, so complaining costs you nothing."))
    return create_model("Grade", __config__=ConfigDict(extra="forbid", strict=True), **fields)


JUDGE_PROMPT = """Grade one answer to an ML research design question. Think briefly.

The scale is calibrated against two anchors. 0.0 is the obvious starting answer the question
rejects: what a competent person writes before they see the problem. 1.0 is the reference answer,
a solution that worked. The rubric points below measure the distance between those two anchors.
The reference is not a ceiling: an answer that beats it scores above 1.0, through the single
`beyond_reference` field and nowhere else. That field is rare and needs you to name what the
reference gets wrong. Every other point is capped at reaching the reference.

The reference answer is *an* answer that worked, not the only one. Credit a point when the answer
reaches the same thing by different means, different notation, or in pseudocode. Do not credit
name-dropping: using the right word while doing something else is a miss. Do not credit hedged
lists that mention the point as one option among many contradictory ones.

The answer is printed with a line number on every line. For each rubric point and each trap, in
this order: set `lines` to the contiguous range that decides it, as "34" or "34-36"; start `rung`
with the rung's numeric value and then name it; then set `score` to the value that rung names. The
rung decides the score. The cited lines are read back from the stored answer and saved verbatim.
A citation that does not exist is discarded with its score. Score each point on its own.

A point that lists its own rungs ("Score 1.0 for ... Score 0.0 for ...") allows those values and
no others: if it names 1.0 and 0.0, there is no 0.5 on it, and an answer that is not a 1.0 is a
0.0. Do not invent a 0.5 compromise after identifying a stated 0.0 case. Only a point with no
rungs takes 0.5, for an answer that makes half the claim.

Leave `lines` and `rung` empty when the point is absent. Any score above zero with no citation is
discarded, so cite or do not claim it. Cite enough complete lines for the evidence to stand on its
own. Judge only the answer text; the reference answer is context, not a target for string matching.

Use one range per point, or two ranges when the point requires two claims that the answer puts in
different places, such as a diagnosis and its repair. Do not cite more than two ranges. Each cited
passage must independently contain one of the required claims.

When prose and pseudocode disagree, the pseudocode wins. An answer that says "one backward pass"
above a loop that takes one backward pass per layer has not made the claim.

The final `ISSUE:` line is an unscored exit interview, not part of the answer. Never cite it,
award a point from it, or fire a trap from it. A random mix of signed coefficients does not prove
that the same batch was evaluated at both signs; require the paired forwards the point asks for.
Likewise, a trap requiring a separate loss needs a separate formula or weighted term, not a generic
prose claim about what the stated loss does.

Most rubric points and traps end with an exclusion: "X does not count", "credit only if", "= miss",
"stops this trap firing". Apply it literally and last, after you have decided you want to credit
the point. If the answer meets the exclusion, `score` is 0.0 however well it does everything else.
This is what `rung` is for. Naming the exclusion there and then scoring above zero is the single
most common way this grading goes wrong, and it is now visible in the record.

One span decides one id. Reusing the same span for two different rubric points means at least one
of them is not really supported. The same holds across a point and a trap: if one span both EARNS
a point and fires a trap, it is one or the other, so decide which and score the other 0.0. This
applies only when the point scored above zero. A span that fails its point is free to fire a trap,
and usually should: the answer that reproduces the baseline the question rejects fails the point
AND falls in the trap.

Put anything odd in `judge_note`: an ambiguous point, a broken item, or an answer that is right in
a way the rubric misses. It is read separately and never scored.

# Question
{prompt}

# Reference answer
{ideal}

# Rubric points
{rubric}

# Traps (a hit here loses marks)
{traps}

# Answer to grade
{answer}

# Required output
Return only one JSON object that matches this exact JSON Schema. Do not use Markdown.
{schema}"""


def _sections(text: str, level: str) -> dict[str, str]:
    """{heading: raw markdown under it} at h2/h3. Fenced code parses as code, so a `## ` inside a
    gold answer's code block is not a heading."""
    lines = text.splitlines()
    tokens = MarkdownIt().parse(text)
    heads = [(tokens[i + 1].content, *t.map) for i, t in enumerate(tokens)
             if t.type == "heading_open" and t.tag == level]
    return {
        head: "\n".join(lines[body_start:heads[j + 1][1] if j + 1 < len(heads) else len(lines)]).strip()
        for j, (head, _, body_start) in enumerate(heads)
    }


def _points(section: str) -> list[dict]:
    """`### <id> (weight <n>)` plus its prose, one per rubric point or trap."""
    points = []
    for head, point in _sections(section, "h3").items():
        pid, weight = re.fullmatch(r"(\w+) \(weight (\d+)\)", head).groups()
        points.append({"id": pid, "weight": int(weight), "point": point + "\n"})
    return points


def load_items(pattern: str = "*.md") -> list[dict]:
    items = []
    for f in sorted(ITEMS.glob(pattern)):
        _, front, body = f.read_text().split("---\n", 2)
        item = yaml.safe_load(front)
        sections = _sections(body, "h2")
        item["prompt"] = sections["prompt"] + "\n"
        assert not _sections(item["prompt"], "h2"), f"{f}: prompt contains an H2 section"
        item["gold_answer"] = sections["gold answer"] + "\n"
        item["rubric"] = _points(sections["rubric"])
        item["traps"] = _points(sections["traps"])
        assert item["id"] == f.stem, f"{f}: id {item['id']!r} does not match filename"
        assert item["rubric"], f"{f}: no rubric points"
        items.append(item)
    return items


# Exit interview, kept out of the score: a confused candidate is evidence about the item. Never
# scored, and it says so, because a candidate that thinks a complaint costs it marks will not
# complain. Friction, not a hedge on the answer: "I was not sure so I did both" is not this.
EXIT_INTERVIEW = (
    "\n\nAfter your answer, add a final line starting with `ISSUE:` if anything about the question "
    "gave you friction: ambiguous or self-contradictory wording, a fact you needed that is not "
    "there and had to guess, numbers that do not add up, or a word limit that will not fit the "
    "answer. Quote the part that caused it. This line is never scored and cannot cost you marks. "
    "Write no ISSUE line if the question was answerable as written."
)


SKILL_HEADER = """You have this reference document open. It is a skill wassname wrote for his own
ML debugging work. Use it if it helps, ignore it if it does not, and do not quote it back at me.

<reference_document>
{text}
</reference_document>

Now the question.

"""


def bench_dataset(pattern: str = "*.md", skill: str | None = None) -> MemoryDataset:
    """The candidate sees the prompt only. The gold answer rides in metadata for the judge.

    `skill` is a markdown file prepended to every question, which makes an uplift variant: the same
    model with a reference document open. Its answers cache under their own scope, so the variant
    and the bare model never serve each other's answers.
    """
    prefix = SKILL_HEADER.format(text=Path(skill).read_text()) if skill else ""
    return MemoryDataset([
        Sample(
            id=item["id"],
            input=prefix + item["prompt"] + EXIT_INTERVIEW,
            target=item["gold_answer"],
            metadata={
                # The question without any uplift document, so both variants are judged on the same
                # text and the judge never pays for the skill the candidate read.
                "question": item["prompt"] + EXIT_INTERVIEW,
                "rubric": item["rubric"],
                "traps": item["traps"],
                "status": item["source"]["status"],
                "title": item["title"],
            },
        )
        for item in load_items(pattern)
    ])


def _numbered(points: list[dict]) -> str:
    return "\n".join(f"- {p['id']} (weight {p['weight']}): {p['point']}" for p in points) or "(none)"


REFUSAL = re.compile(r"\b(I can't|I cannot|I won't|I'm unable|as an AI)\b", re.I)
ANSWER_UNDER_TEST = "the answer under test, written out at length"


LINES = re.compile(r"(\d+)\s*(?:[-–]\s*(\d+))?")
MAX_CITED = 30


def with_line_numbers(answer: str) -> str:
    """The answer as the judge sees it, one number per line, so a hit cites lines not prose."""
    return "\n".join(f"{i:4d}| {line}" for i, line in enumerate(answer.splitlines(), 1))


def cited_lines(answer: str, spec: str) -> str:
    """The text of the cited lines. Empty when the citation is missing or out of range.

    The judge writes "34", "34-36" and "12,22", so a spec is a set of ranges rather than one
    range. Only the lines named are returned, never the gap between two of them, or citing
    12,22 would hand it every line in between to quote from.
    """
    lines = answer.splitlines()
    cited = {n for first, last in LINES.findall(spec)
             for n in range(int(first), int(last or first) + 1)}
    return "\n".join(lines[n - 1] for n in sorted(cited)[:MAX_CITED] if 0 < n <= len(lines))


def _cited(point_id: str, check: "PointCheck", answer: str, rejected: list[str]) -> bool:
    """A hit must cite one or two real ranges of the stored answer."""
    if 0 < len(LINES.findall(check.lines)) <= 2 and cited_lines(answer, check.lines):
        return True
    rejected.append(f"{point_id}: lines {check.lines!r}")
    return False


def _shared_quotes(grades: list, rubric: list[dict], answer: str) -> list[str]:
    """Ids that two or more credited points rest on the same span for.

    Pass rubric + traps: one span that both earns a point and fires a trap is the loudest
    version of this, and it means the trap is catching the legitimate neighbour.

    Nesting counts. Exact string equality missed three real double-charges in one round, where
    the judge quoted a sentence for one id and a prefix of it for another.
    """
    shared = []
    for grade in grades:
        credited = [(point["id"], " ".join(cited_lines(
                        answer, getattr(grade, point["id"]).lines).split()))
                    for point in rubric
                    if getattr(grade, point["id"]).score > 0
                    and cited_lines(answer, getattr(grade, point["id"]).lines)]
        for i, (point_id, span) in enumerate(credited):
            for other_id, other in credited[i + 1:]:
                if span in other or other in span:
                    shared.append(f"{point_id} + {other_id}")
    return sorted(set(shared))


def _on_anchors(judge: str, score: float) -> float:
    """Put one judge's score on the shared scale, so panel members are averageable.

    Every judge is measured on the same two fixed artifacts by `just calibrate`: the item's own
    gold answer, which must read 1.0, and another item's gold answer, which must read 0.0. Mapping
    each judge through its own two anchors keeps the meaning of 0 and 1 and, unlike a z-score,
    does not depend on which models are in the table, so a judge sitting out its own family cannot
    move anyone's number and the month-to-month comparison survives.
    """
    off, gold = JUDGE_ANCHORS[judge.removeprefix("openrouter/")]
    return (score - off) / (gold - off)


def _sits_out(judge: str, candidate: str) -> bool:
    """A seat does not grade its own company. On v93 luna scored openai answers +0.02 above the
    rest of the field, deepseek-v4-pro did the same for deepseek, and luna grading luna was +0.10
    (scripts/scratch/judge_self_bias.py).
    """
    return _company_of(judge) == _company_of(candidate)


def _seat_effects(judgments: list[tuple[str, str, float]]) -> dict[str, float]:
    """How lenient each seat is in the middle of the scale, where the anchors do not reach.

    `judgments` is (answer key, seat, anchored score). Fits `y = mu[answer] + delta[seat]` by
    alternating means, which is the two-way fit with the missing cells the dropout creates, and
    centres delta to sum to zero. A seat's delta is what it adds to any answer it grades, so the
    shift a row takes from its own seat sitting out is the mean delta of the seats that did sit.
    """
    delta = {seat: 0.0 for _, seat, _ in judgments}
    by_answer: dict[str, list[tuple[str, float]]] = {}
    for answer, seat, y in judgments:
        by_answer.setdefault(answer, []).append((seat, y))
    for _ in range(200):
        mu = {a: statistics.mean(y - delta[s] for s, y in rows) for a, rows in by_answer.items()}
        residual: dict[str, list[float]] = {seat: [] for seat in delta}
        for a, rows in by_answer.items():
            for seat, y in rows:
                residual[seat].append(y - mu[a])
        delta = {seat: statistics.mean(r) for seat, r in residual.items()}
        centre = statistics.mean(delta.values())
        delta = {seat: d - centre for seat, d in delta.items()}
    return delta


def _judge_spread(judgments: list[tuple[str, str, float]],
                  effects: dict[str, float]) -> dict[str, float]:
    """How much two seats still disagree about one answer, once the scale is taken out.

    Both anchors and leniency are removed first, so this is not "judges use different scales", it
    is what is left after correcting for that: the same answer read two ways. `sd` is per judgment,
    `se` is what that leaves on a model's row of `n` judgments, and two rows closer than `se` are
    not separated by this bench.
    """
    if not effects:
        return {}
    by_answer: dict[str, list[tuple[str, float]]] = {}
    for answer, seat, y in judgments:
        by_answer.setdefault(answer, []).append((seat, y))
    residuals = []
    for rows in by_answer.values():
        mu = statistics.mean(y - effects[s] for s, y in rows)
        residuals += [y - effects[s] - mu for s, y in rows]
    sd = statistics.pstdev(residuals)
    # The key is "<variant>/<item>", and the variant itself holds slashes, so the model is
    # everything
    # before the last one.
    per_row = len(judgments) / len({a.rsplit("/", 1)[0] for a, _, _ in judgments})
    return {"sd": round(sd, 3), "judgments_per_row": round(per_row),
            "se": round(sd / math.sqrt(per_row), 3)}


def _judge_id(judge_model) -> str:
    """One string naming who graded, so a table never mixes a panel with a single judge."""
    if isinstance(judge_model, tuple):
        return "+".join(sorted(m.removeprefix("openrouter/") for m in judge_model))
    return str(judge_model).removeprefix("openrouter/")


def _company_of(model: str) -> str:
    """Vendor slug, from any of `openrouter/deepseek/x`, `deepseek/x` or `mockllm/judge`."""
    return model.removeprefix("openrouter/").split("/")[0]


@scorer(metrics={"score": [mean(), stderr()], "trapped": [accuracy()], "answered": [accuracy()]})
def rubric_judge(panel: tuple[str, ...] | Model = JUDGE_PANEL, passes: int = JUDGE_PASSES,
                 temperature: float = JUDGE_TEMPERATURE,
                 max_tokens: int = JUDGE_MAX_TOKENS):
    """Weighted rubric hits minus weighted traps, averaged over the panel.

    `passes` samples from each panel member, so the score averages over both sampling noise and
    judge choice. A judge sits out when it shares a company with the model it would grade: on v93,
    luna scored openai answers +0.02 above the rest of the field and deepseek-v4-pro did the same
    for deepseek, and luna grading luna was +0.10 (scripts/scratch/judge_self_bias.py).

    The mean is plain, not standardised. A z-score is taken over whichever models happen to be in
    the table, so adding three Qwen sizes would move everyone else's number and the month-to-month
    drift check would break. The measured per-judge offset is only 0.012 anyway, against 0.104 of
    judge-by-model disagreement, so standardising would fix the small part and cost the scale.
    """
    if isinstance(panel, Model):  # the smoke test's mock judge
        judges, quiet = [panel], {}
    else:
        # A bare string is one judge, not five one-character judges. `just calibrate` measures one
        # seat at a time and passes its name straight through, and iterating that string asked
        # OpenRouter for a model called "o".
        names = (panel,) if isinstance(panel, str) else panel
        judges = [_openrouter_model(name) for name in names]
        # Resolved here, not in the retry, so the offline smoke never reaches OpenRouter for a mock
        # judge that has no model record.
        quiet = {j.name: _lowest_effort(n) for j, n in zip(judges, names)}

    async def score(state: TaskState, target: Target) -> Score:
        rubric: list[dict] = state.metadata["rubric"]
        traps: list[dict] = state.metadata["traps"]
        answer = state.output.completion
        # A 0.00 from an empty, refused, or provider-blocked answer is a harness failure, not a
        # model result. opus-5 via OpenRouter returns content_filter on three items with a
        # 206-character policy notice, which scores 0 and outranks nothing.
        # Still truncated after the solver's continuation retry, so the answer is a fragment and
        # every later rubric point scores 0 for a reason that has nothing to do with the model.
        # Three judge passes independently called this out on lucid_review before it was caught.
        truncated = state.output.stop_reason == "max_tokens"
        answered = (
            bool(answer.strip())
            and not truncated
            and state.output.stop_reason != "content_filter"
            and not REFUSAL.search(answer[:400])
        )
        if not answered:
            return Score(
                value={"score": 0.0, "trapped": 0.0, "answered": 0.0},
                answer=answer,
                explanation=f"no gradable answer: stop_reason={state.output.stop_reason}",
                metadata={"points": {}, "traps": {}, "judge_notes": [], "quotes": {},
                          "truncated": truncated},
            )
        schema = grade_model(rubric, traps)
        prompt = JUDGE_PROMPT.format(
            prompt=state.metadata["question"],
            ideal=target.text,
            rubric=_numbered(rubric),
            traps=_numbered(traps),
            answer=with_line_numbers(answer),
            schema=json_schema(schema).model_dump_json(exclude_none=True),
        )
        # Reasoning on, no separate budget: `max_tokens` already bounds thinking plus the JSON,
        # and thinking is most of the reply anyway. With reasoning off the judge did its thinking
        # in `judge_note`, which is emitted after every score is written, so it kept naming the
        # right rung and recording the wrong score: six of thirteen items in round 20. An empty
        # reply (a model spending every token thinking) is handled by the retry in `_generate_grade`.
        config = GenerateConfig(
            temperature=temperature,
            max_tokens=max_tokens,
            response_schema=ResponseSchema(name="grade", strict=True,
                                          json_schema=json_schema(schema)),
            max_retries=MAX_RETRIES,
            # Each seat is its own provider, so 16 each is 16 per provider. The seats are what a
            # whole-roster sweep queues on: every candidate's every sample needs all five before
            # the sample counts, and inspect takes the connection slot before it reads the cache,
            # so a short judge pool holds up answers that were already graded.
            max_connections=JUDGE_MAX_CONNECTIONS,
        )
        # Both the family rule and the anchor map are properties of a panel. A lone judge keeps
        # its raw score: it has nobody to sit out for and nobody to be made comparable to, and the
        # calibration run that measures the anchors must not be scored through them.
        panelled = len(judges) > 1
        sitting = [j for j in judges
                   if not panelled or not _sits_out(j.name, str(state.model))]
        graded_by = [(j, p) for j in sitting for p in range(passes)]
        # All at once. Each seat is a different company, so five in flight is one request per
        # provider, not five against one. Serial here made the panel five times slower than a
        # single judge and left four providers idle the whole time. gather keeps the order, so
        # `graded_by` still names who wrote which grade.
        # A seat that fails after both its retries drops out of this cell instead of killing it.
        # One seat of five moves a cell by less than the seats already disagree, while a dead cell
        # drops the whole model from the table: deepseek-v4-flash-0731 spent every token thinking
        # on CW#2 and would have taken claude-sonnet-5 out of v96. Over half the
        # panel failing is a harness fault, not a grade, and still raises.
        results = await asyncio.gather(
            *(_grade(j, prompt, config, p, schema, quiet.get(j.name, {}))
              for j, p in graded_by),
            return_exceptions=True)
        failed = [f"{j.name}: {type(e).__name__}: {e}"[:300]
                  for (j, _), e in zip(graded_by, results) if isinstance(e, BaseException)]
        graded_by = [pair for pair, g in zip(graded_by, results)
                     if not isinstance(g, BaseException)]
        grades = [g for g in results if not isinstance(g, BaseException)]
        if not _panel_survives(len(grades), len(results)):
            raise RuntimeError(f"{len(failed)} of {len(results)} judge passes failed: {failed}")

        uncited: list[str] = []

        def credit(point_id: str, check, rejected: list[str]) -> float:
            """Clamped score, zeroed unless it cites a real answer line."""
            score = _consistent_score(check)
            return score * _cited(point_id, check, answer, rejected) if score else 0.0

        def rate(point_id: str) -> float:
            """Mean over passes. The schema makes every id a required field, so none can vanish."""
            return statistics.mean(
                credit(point_id, check, uncited)
                for check in (getattr(grade, point_id) for grade in grades)
            )

        total = sum(p["weight"] for p in rubric)

        def score_of(grade) -> float:
            """What this item would score on one judge pass alone."""
            def one(point_id: str) -> float:
                return credit(point_id, getattr(grade, point_id), [])

            return ((sum(p["weight"] * one(p["id"]) for p in rubric)
                     - sum(p["weight"] * one(p["id"]) for p in traps)) / total
                    + BEYOND_WEIGHT * one(BEYOND_ID))

        shared = _shared_quotes(grades, rubric + traps, answer)
        lost = sum(p["weight"] * rate(p["id"]) for p in traps)
        beyond = BEYOND_WEIGHT * rate(BEYOND_ID)
        # Each judge's own score, put on the shared scale, then averaged. Averaging the raw scores
        # and averaging the per-point rates give the same number, since the combination is linear,
        # so the only change here is the per-judge anchor map.
        per_judge = [_on_anchors(j.name, score_of(g)) if panelled else score_of(g)
                     for (j, _), g in zip(graded_by, grades)]
        return Score(
            value={"score": statistics.mean(per_judge),
                   "trapped": float(lost > 0), "answered": 1.0},
            answer=answer,
            explanation=grades[0].evidence,
            metadata={
                "points": {**{p["id"]: rate(p["id"]) for p in rubric}, BEYOND_ID: beyond},
                "traps": {p["id"]: rate(p["id"]) for p in traps},
                "judge_notes": [g.judge_note for g in grades if g.judge_note],
                # Traps too: without their quotes an auditor cannot check a trap that fired, which
                # is how `alpha_sampling_only` kept firing on an answer its own text excludes.
                # every pass, not just the first: the passes disagree, and three audits in a row
                # had to open the raw .eval because a mean of 0.33 hides which pass voted why
                "quotes": {p["id"]: [(getattr(g, p["id"]).score, getattr(g, p["id"]).lines,
                                      cited_lines(answer, getattr(g, p["id"]).lines),
                                      getattr(g, p["id"]).rung)
                                     for g in grades]
                           for p in rubric + traps + [{"id": BEYOND_ID}]},
                "rung_mismatches": {
                    p["id"]: [(i, getattr(g, p["id"]).score, getattr(g, p["id"]).rung)
                              for i, g in enumerate(grades)
                              if not _score_matches_rung(getattr(g, p["id"]))]
                    for p in rubric + traps
                    if any(_rung_score(getattr(g, p["id"]).rung)
                           != getattr(g, p["id"]).score for g in grades)
                },
                "truncated": truncated,
                # Credits refused for citing no real answer line.
                "uncited": sorted(set(uncited)),
                # Two rubric points resting on the same span: one of them is being padded.
                "shared_quotes": shared,
                # What each pass alone would have scored. The spread over these is judge noise,
                # and a defect that moves a score less than this is not worth an item revision.
                # RAW, unlike `value["score"]`, which is the mean of these put through each seat's
                # anchors. Anything comparing the two must call `_on_anchors` first, as `_results`
                # does for the leniency fit. Anchoring them here would be tidier and is a v95 job,
                # not a mid-sweep one: it would leave this sweep's logs half raw and half anchored
                # with nothing in the file saying which. `points` and `traps` below are raw too.
                "pass_scores": [score_of(g) for g in grades],
                # Which panel member wrote each of those, so disagreement is attributable and a
                # judge that is always the outlier can be replaced on evidence.
                "graded_by": [j.name for j, _ in graded_by],
                # Named here rather than inferred, because a judge that sat out for sharing the
                # candidate's company is not the same thing as a judge that failed.
                "sat_out": [j.name for j in judges if j not in sitting],
                # A seat that dropped out because it errored. Empty on a healthy cell. A seat that
                # appears here often is failing on a kind of answer and wants replacing.
                "failed_seats": failed,
            },
        )

    return score


def _panel_survives(ok: int, asked: int) -> bool:
    """Can a cell still be scored after `asked - ok` of its judge passes failed? Over half gone is
    a harness fault, and a lone judge failing takes the cell with it."""
    return ok * 2 > asked - 1


def _reasoning_text(output) -> str:
    """Some routes put the whole JSON answer in the reasoning channel. Read it rather than lose it."""
    content = output.message.content
    parts = content if isinstance(content, list) else []
    return "".join(p.reasoning for p in parts if getattr(p, "reasoning", None))


async def _grade(
    judge: Model,
    prompt: str,
    config: GenerateConfig,
    pass_index: int,
    schema: type[BaseModel],
    quiet: dict,
) -> BaseModel:
    """One graded pass. Every rubric id is a required field of `schema`, so none can go missing."""
    return await _retry_stream_error(
        lambda: _generate_grade(judge, prompt, config, pass_index, schema, quiet))


async def _generate_grade(
    judge: Model, prompt: str, config: GenerateConfig, pass_index: int,
    schema: type[BaseModel], quiet: dict, cache=...
) -> BaseModel:
    cache = _cache(pass_index, judge.name) if cache is ... else cache
    output = await judge.generate(prompt, config=config, cache=cache)
    if not output.completion.strip() and _reasoning_text(output).lstrip().startswith("{"):
        return _validate(schema, _reasoning_text(output))
    if not output.completion.strip():
        output = await judge.generate(
            [
                ChatMessageUser(content=prompt),
                output.message,
                ChatMessageUser(content="Continue grading the numbered candidate answer in the "
                                        "first user message. Return only the required JSON now."),
            ],
            # An empty reply means the judge spent every token thinking, so the retry asks for less
            # thinking and gives it 3x the room. Turning reasoning off outright is what gemini 400s
            # on, and deepseek thinks past a 4k cap even at its lowest rung, so neither alone works.
            config=config.merge(GenerateConfig(reasoning_effort=None,
                                               extra_body=quiet,
                                               max_tokens=config.max_tokens * 3)),
            cache=cache,
        )
    if not output.completion.strip():
        raise RuntimeError(
            f"judge returned no text twice: stop_reason={output.stop_reason} "
            f"usage={output.usage} message={output.message.content!r:.400}\n"
            # Both rungs above read from the cache, which never expires, so re-running replays the
            # same empty reply and this cell stays dead. Clearing one seat re-pays that seat's
            # grading only; the candidates' answers live in their own model caches.
            f"This cell cannot retry itself out. To unfreeze it: "
            f"inspect cache clear --model {judge.name}"
        )
    try:
        return _validate(schema, output.completion)
    except ValidationError as error:
        # Return the invalid JSON and exact error, or a fresh retry often repeats the same fault.
        output = await judge.generate(
            [
                ChatMessageUser(content=prompt),
                output.message,
                ChatMessageUser(content=f"Your JSON failed validation:\n{error}\n"
                                        "Correct the JSON only. Keep every grading decision unless the "
                                        "error shows it is internally inconsistent."),
            ],
            # Cached like the other two rungs. The invalid reply it is correcting came from the
            # cache itself, so the whole ladder replays. Uncached, a seat that fails validation
            # re-paid this call on every re-run: 194k tokens of one replay of seven finished
            # models, all of it inkling-small and qwen3.7-flash correcting themselves again.
            cache=cache,
            config=config.merge(GenerateConfig(
                temperature=1.0,
                reasoning_effort=None,
                extra_body=quiet,
                max_tokens=config.max_tokens * 3,
            )),
        )
        if not output.completion.strip():
            raise RuntimeError(
                f"judge returned no text on validation retry: stop_reason={output.stop_reason} "
                f"usage={output.usage} message={output.message.content!r:.400}\n"
                f"This cell cannot retry itself out. To unfreeze it: "
                f"inspect cache clear --model {judge.name}"
            )
        return _validate(schema, output.completion)


def _validate(schema: type[BaseModel], text: str) -> BaseModel:
    """The provider does not enforce the strict schema it is sent, so the judge sometimes writes a
    `score_justification` next to the cited lines and score. Extra keys carry no credit.

    A fenced reply is the same JSON with ```json around it, and it cost claude-haiku-4.5 a whole
    cell on VJ#12 at v96. Unfencing here is not the same as tolerating a bad grade: the fence
    is formatting, and what is inside still has to pass the strict schema.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return schema.model_validate_json(text, extra="ignore")


def _cache(pass_index: int, judge: str) -> CachePolicy:
    # Each pass caches separately, so a rerun replays the same N samples instead of paying again.
    # Scope the judge too, or a re-run with a different judge replays the old judge's scores.
    return CachePolicy(expiry=None, scopes={"rubric": RUBRIC_VERSION, "pass": str(pass_index), "judge": judge})


@hooks(name="sweep_progress", description="one bar over every answer in the sweep")
class _SweepProgress(Hooks):
    """One tqdm bar over every (model, item), counted and ticked by inspect's own events.

    The bar is a class attribute rather than a module global because inspect imports bench.py a
    second time to resolve the task, and the copy that holds the registered hook is not the copy
    that `__main__` set a global on. The parallel sweep has no outer loop to count either, so the
    total grows as tasks start.

    mininterval stops a redraw per sample, and maxinterval keeps writing a line while a slow model
    thinks, which is what tells a tailed log the run is alive rather than hung.
    """

    bar = None

    async def on_task_start(self, data: TaskStart) -> None:
        from tqdm.auto import tqdm

        if _SweepProgress.bar is None:
            _SweepProgress.bar = tqdm(total=0, unit="answer", smoothing=0.0,
                                      mininterval=5.0, maxinterval=120.0)
        _SweepProgress.bar.total += data.spec.dataset.samples or 0
        _SweepProgress.bar.refresh()

    async def on_sample_end(self, data: SampleEnd) -> None:
        _SweepProgress.bar.update(1)

    async def on_task_end(self, data: TaskEnd) -> None:
        if data.log:
            _SweepProgress.bar.write(f"done {data.log.eval.model} status={data.log.status}")

    async def on_run_end(self, data: RunEnd) -> None:
        _SweepProgress.bar.close()
        _SweepProgress.bar = None


@task
def wassname_ml_bench(
    items: str = "*.md",
    judge_model: str | Model | tuple[str, ...] = JUDGE_PANEL,
    judge_passes: int = JUDGE_PASSES,
    judge_temperature: float = JUDGE_TEMPERATURE,
    # Already resolved to a rung the model lists, by `_lowest_effort` in the caller, because a
    # batch shares one GenerateConfig and the lowest rung differs per model.
    reasoning: dict = {},
    # What the continuation turn sends, from `_quiet_effort` in the caller for the same reason.
    # Resolved there and not in the solver so the offline smoke never reaches OpenRouter.
    quiet: dict = {},
    judge_max_tokens: int = JUDGE_MAX_TOKENS,
    epochs: int = 1,
    cache_scope: str | None = None,
    skill: str | None = None,
    draw: int = 1,
    provider: dict | None = None,
    max_tokens: int = ANSWER_MAX_TOKENS,
    # None means the default arm, this model's lowest rung, which files in the bare row. A named
    # level is an override and gets its own row, because a model told to think harder is a
    # different candidate.
    effort_label: str | None = None,
) -> Task:
    # An uplift variant is its own row in the table, keyed on this scope, so it never shares a
    # cached answer with the bare model.
    # A non-default answer budget is a variant for the same reason: an answer written under a
    # thinking budget is a different answer, and without this it would replay the 40k cache.
    variant = ([f"skill:{Path(skill).parent.name}"] if skill else []) + (
        [] if effort_label is None else [f"effort:{effort_label}"]) + (
        [] if max_tokens == ANSWER_MAX_TOKENS else [f"budget:{max_tokens}"])
    cache_scope = cache_scope or " ".join(variant) or None
    return Task(
        dataset=bench_dataset(items, skill),
        solver=generate_tolerating_content_filter(quiet),
        scorer=rubric_judge(judge_model, judge_passes, judge_temperature, judge_max_tokens),
        epochs=Epochs(epochs, "mean"),
        config=GenerateConfig(
            max_tokens=max_tokens,
            max_connections=MAX_CONNECTIONS,
            max_retries=MAX_RETRIES,
            extra_body=reasoning | (provider or {}),
            # No rubric scope: an answer does not depend on the rubric, and scoping it here made
            # every rubric edit re-roll all the answers too. -- CLAUDE
            # The draw index lives here and not in the request. As `seed` it would have been one
            # more parameter for require_parameters to insist on, and Anthropic does not take a
            # seed, so the whole company would have had no allowed provider.
            # One week, not forever: a monthly re-run has to re-ask the model, or a vendor cutting
            # thinking effort or quantising a model would be invisible behind a replayed answer.
            # The .eval logs are the history and are never served from here.
            cache=CachePolicy(
                expiry="1W",
                # "arm" is a stored cache key, not a name, so it keeps its old spelling: renaming it
                # to match the code would re-ask every variant row for nothing.
                scopes={"draw": str(draw)} | ({} if cache_scope is None else {"arm": cache_scope})),
        ),
        metadata={"rubric_version": RUBRIC_VERSION, "judge": _judge_id(judge_model),
                  "cache_scope": cache_scope, "draw": draw,
                  # The rung these answers actually ran at. The arm is "lowest listed", which is a
                  # different level per model, so the report cannot name it from a constant.
                  "effort": reasoning.get("reasoning", {}).get("effort"),
                  # Which serving policy produced these answers, so a table never mixes two.
                  "provider": json.dumps((provider or {}).get("provider", {}), sort_keys=True)},
        fail_on_error=0.1,
        continue_on_fail=True,
    )


def _trace(log_file: str, sample_id: str | None = None) -> None:
    """Print one item in full: prompt, answer, grade. Read this before believing any number."""
    from inspect_ai.log import read_eval_log

    log = read_eval_log(log_file)
    samples = [s for s in log.samples if sample_id in (None, s.id)]
    sample = samples[0]
    print(f"# {sample.id} epoch={sample.epoch} model={log.eval.model}")
    for message in sample.messages:
        print(f"\n## {message.role}\n{message.text}")
    if sample.error:
        print(f"\n## error\n{sample.error}")
        return
    grade = sample.scores["rubric_judge"]
    print(f"\n## grade\n{grade.value}\n{grade.explanation}")
    if not grade.value.get("answered", 1.0):
        print("\nCHECK: no grade exists because the candidate answer was incomplete.")
        return
    _print_grade(grade)
    print("\nCHECK: the answer must be a real attempt at THIS question, not a refusal, not a "
          "restatement of the prompt, not truncated. Every hit needs a quote that supports it.")


def _graded_logs(log_dir: str, any_version: bool = False, any_status: bool = False,
                 with_usage: bool = False):
    """Successful graded logs under `log_dir`, every judge, at the current rubric version.

    `any_version` is for token accounting, where an older run is still a valid cost profile.
    `any_status` is for spend: a run that died partway still burned tokens, and dropping it
    priced a re-run model from its last attempt alone (claude-fable-5 read $0.005 against ~$1.08).
    Each log carries its key (model, judge, version) in metadata; callers dedupe on that key,
    not on the directory layout.
    """
    from inspect_ai.log import list_eval_logs, read_eval_log

    for info in list_eval_logs(log_dir):
        # Header first. A body is most of a megabyte and the callers throw most of them away, so
        # reading every body under logs/ turned the report into a several minute job.
        head = read_eval_log(info.name, header_only=True)
        if head.status != "success" and not any_status:
            continue
        if not any_version and head.eval.metadata.get("rubric_version") != RUBRIC_VERSION:
            continue
        # A run whose answers all came from cache billed nobody but the judges, so a caller that
        # prices candidates has nothing to read in its body. Most logs under logs/ are that.
        if with_usage and not any(not _is_judge(billed, head) for billed in head.stats.model_usage):
            continue
        log = read_eval_log(info.name)
        if log.status != "success" and not any_status:
            continue
        if not any_version and log.eval.metadata.get("rubric_version") != RUBRIC_VERSION:
            continue
        yield log


# The lowest effort a model will accept, where that is above the bench's own setting. o1 rejects
# `minimal` ("Supported values are: 'low', 'medium', and 'high'"), so its floor is its bare row.
# Without this it has only a variant row, no row to lift against, and no place on the timeline.
EFFORT_FLOOR = {"openrouter/openai/o1": "low"}


def _variant(log) -> str:
    """The key a score is reported under: the model, plus the uplift variant when it ran with one.

    `openrouter/deepseek/deepseek-v4-flash-0731 (skill:ml-debug)` is a different candidate from the
    bare model, so it gets its own row, its own cost and its own cached answers. A second draw is
    not a variant: it is the same candidate answering again, and it averages into the same row.
    Calibration scopes are not variants: those rows are keyed on the mock model instead.
    """
    scope = log.eval.metadata.get("cache_scope") or ""
    if scope == f"effort:{EFFORT_FLOOR.get(log.eval.model)}":
        return log.eval.model
    return (f"{log.eval.model} ({scope})"
            if scope.startswith(("skill:", "effort:", "budget:")) else log.eval.model)


def _draw_of(log) -> int:
    """Which of the N draws of the same model this log holds.

    The draw index is the seed, so each draw stores and replays as its own answer instead of
    overwriting the last. The first draws were run before the seed existed and carry the index in
    their cache scope instead, and they cost real money, so both spellings are read here rather
    than thrown away. Nothing else in the file needs to know there are two.
    """
    meta = log.eval.metadata or {}
    scope = meta.get("cache_scope") or ""
    if scope.startswith("repeat:"):
        return int(scope.removeprefix("repeat:"))
    return int(meta.get("draw") or 1)


def _model_of(variant: str) -> str:
    """The model id back out of a variant key, for a price or company lookup."""
    return variant.split(" (")[0]


def _short_variant(variant: str) -> str:
    """Row label: model name only, keeping the variant suffix. The company is the marker colour."""
    model, _, rest = variant.partition(" (")
    return model.split("/")[-1] + (f" ({rest}" if rest else "")


# A row label carrying a variant, `deepseek-v4-flash-0731 (effort:high)`. `(with fallback)` is not
# a variant: that row is still the bare model, reading a sibling's score where its provider refused.
_VARIANT_LABEL = re.compile(r" \((skill|effort|budget):")


# A reader cannot judge a skill's lift without reading the skill, so the label links to it.
_SKILL_LINK = {"skill:ml-debug": "[skill:ml-debug](https://github.com/wassname/ml-debug)"}


def _arm_scope(row: dict) -> dict:
    """The three knobs an arm can differ on, read back out of its row label."""
    scope = dict(effort=EFFORT_ARM, budget=ANSWER_MAX_TOKENS, skill=None)
    for token in row["model"].partition(" (")[2].rstrip(")").split():
        key, _, value = token.partition(":")
        scope[key] = int(value) if key == "budget" else value
    return scope


def _lift(a: dict, b: dict, items: list[str]) -> dict:
    """The error bar on b minus a, paired by question over the 12 both answered.

    The change itself is not returned: it is the difference of two scores the table already prints.

    Never paired by draw: the draw index is a cache scope, not a seed, so draw 3 of one arm is not
    the partner of draw 3 of the other, and the pairing chosen would decide the sign (AGENTS.md 6).
    """
    diffs = [b[i] - a[i] for i in items]
    return {"+-": statistics.stdev(diffs) / math.sqrt(len(diffs))}


def _thinking(row: dict) -> float | None:
    """Reasoning tokens for one answer, which is what a reasoning rung is supposed to move."""
    share = row.get("reasoning share")
    return None if share is None or row["tok/answer"] is None else row["tok/answer"] * share


# The ladder a model climbs, each rung one change from the rung above it.
_RUNGS = [("low effort", dict(skill=None, effort=EFFORT_ARM)),
          ("high effort", dict(skill=None, effort="high")),
          ("high effort + skill:ml-debug", dict(skill="ml-debug", effort="high"))]


def _ladder(mine: list[tuple[dict, dict]], budget: int) -> list[tuple[str, dict]]:
    """The rungs this model has at this budget, lowest first, dropping any the provider ignored."""
    rungs = [(label, [row for row, s in mine if s["budget"] == budget
                      and all(s[k] == v for k, v in want.items())])
             for label, want in _RUNGS]
    rungs = [(label, found[0]) for label, found in rungs if found]
    # A rung the provider did not honour is not a comparison. deepseek's lowest listed rung thinks
    # to the cap like its highest, 11,832 tokens against 12,444, so its "low" arm is the same
    # candidate as its "high" one and only the budget ever controlled it. Grok's rungs are real,
    # 1,169 against 6,788. Threshold on thinking, not on the label.
    if len(rungs) > 1 and rungs[0][0] == "low effort":
        thinking = [_thinking(arm) for _, arm in rungs[:2]]
        if all(thinking) and thinking[1] < 1.2 * thinking[0]:
            rungs = rungs[1:]
    return rungs if len(rungs) > 1 else []


def _uplift(variants: list[dict], base: list[dict], items: list[str]) -> list[dict]:
    """One row per arm, in ladder order, so `lift` reads against the row above it.

    The whole section runs at one thinking budget, the largest any model has a ladder at, and a
    model with no ladder there is left out rather than compared across budgets. A document read
    under a budget the model was already spending on thinking measures crowding rather than the
    document (logs_capbound/README.md), so the budget has to be held fixed and worth stating once
    in the prose instead of carried as a column.
    """
    arms = [(row, _arm_scope(row)) for row in variants + base if row["score"] is not None]
    by_model = {model: [(row, s) for row, s in arms
                        if row["model"].partition(" (")[0] == model]
                for model in {row["model"].partition(" (")[0] for row, _ in arms}}
    budgets = sorted({s["budget"] for _, s in arms}, reverse=True)
    budget = next((b for b in budgets if any(_ladder(mine, b) for mine in by_model.values())), None)
    groups = [[{"model": model, "arm": label, "score": arm["score"],
                # The first rung is the thing being lifted from, so it has no lift of its own.
                **({"+-": None} if i == 0 else _lift(ladder[i - 1][1], arm, items)),
                "tok/answer": arm["tok/answer"]}
               for i, (label, arm) in enumerate(ladder)]
              for model, mine in by_model.items() if (ladder := _ladder(mine, budget))]
    # Models by their best rung, rungs in ladder order inside a model.
    return [row for group in sorted(groups, key=lambda g: -max(r["score"] for r in g))
            for row in group]


def _per_answer(tokens: int | None, n_items: int) -> float | None:
    """Tokens for one answer. `tokens` is the model's bill for the whole run of n_items."""
    return None if tokens is None else tokens / n_items


def _millions(tokens: int | None) -> float | None:
    return None if tokens is None else tokens / 1_000_000


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or not denominator:
        return None
    return numerator / denominator


def _fallback_note(borrow: tuple[str, set[str]] | None, n_items: int) -> str | None:
    """Hover text for a row that reads a sibling's score on the questions its provider blocked."""
    if not borrow:
        return None
    donor, gaps = borrow
    return f"with fallback to {_short_variant(donor)} for {len(gaps)}/{n_items} refused questions"


def _ktok_text(row: dict) -> str:
    """Tokens for the hover card. Unknown when every answer came from the cache, which records
    no usage at all."""
    if row["tok/answer"] is None:
        return "tokens unknown, answers served from cache"
    share = row.get("reasoning share")
    return (f"{row['tok/answer']:,.0f} tokens per answer, reasoning included, against a requested "
            f"{ANSWER_MAX_TOKENS:,}-token thinking budget"
            + (f"<br>reasoning effort '{EFFORT_ARM}', and "
               f"{share:.0%} of the output was reasoning" if share is not None else ""))


def _mean_if_complete(values, model: str, incomplete: dict[str, int], scale: float = 1.0):
    """None for a model that skipped questions, so the aggregate cell renders blank.

    A mean over 10 of 12 questions is not comparable to a mean over 12, but the per-item cells the
    model did answer are still worth reading, so the row stays.
    """
    values = list(values)
    if model in incomplete or not values:
        return None
    return statistics.mean(values) * scale


def _components(rows: dict[str, dict[str, list[float]]], spread: dict | None,
                n_items: int) -> dict:
    """Cut one row's error bar into the three things that move a score.

    A cell is one answer to one question, read by the panel, so its spread already holds all
    three. This is a variance components split of that one bar, not three bars added: the same
    facet decomposition generalizability theory uses for raters and items.

        between answers   from a `repeat:` variant, the same model answering again
        between judges    from the panel's own disagreement on one answer
        between questions whatever is left, which is real difficulty

    Needs a repeat variant, so it returns nothing until one has run.

    SHOULD: the three add in quadrature to about the row's `+-`. ELSE a term is double counted,
    which is the mistake this replaced.
    """
    out = {}
    for variant, items in rows.items():
        # Pooled over questions, so five draws of twelve questions give 48 degrees of freedom
        # rather than 4. A question answered once contributes nothing and is skipped.
        per_item = [statistics.stdev(v) for v in items.values() if len(v) > 1]
        if len(per_item) < n_items / 2:
            continue
        answers = math.sqrt(statistics.mean(s ** 2 for s in per_item))
        judges = spread["sd"] / math.sqrt(spread["judgments_per_row"] / n_items) if spread else 0.0
        cells = statistics.pstdev([statistics.mean(v) for v in items.values()])
        # The fewest draws any question got, matching the `draws` column. A row cut short partway
        # through a draw has 2 answers to some questions and 1 to others, and taking the most would
        # over-subtract below.
        k = min(len(v) for v in items.values())
        # A cell that averages k draws already holds only a 1/k share of the answer variance, so
        # subtracting a whole draw's worth would charge the question term for noise the average
        # has removed and read low. The three parts below are all per draw.
        questions = math.sqrt(max(cells ** 2 - answers ** 2 / k - judges ** 2, 0.0))
        out[variant] = {"draws": k, "total": math.hypot(math.hypot(questions, answers), judges),
                    "achieved": cells,
                    "sd": [("between questions", questions), ("between answers", answers),
                           ("between judges", judges)]}
    return out


def _item_se(values, model: str, incomplete: dict[str, int]):
    """How far the row mean would move on another 12 questions of the same kind.

    Each item score is already an average over judge seats and over the draws the model answered,
    so this carries the judge noise, the re-answering noise divided by the draws, and the much
    larger item-to-item spread. `_components` is what splits those three apart.

    SHOULD: read 0.06 to 0.09, three to four times the judge-only 0.019. ELSE the items agree
    with each other more than the seats do, which no run has shown yet.
    """
    values = list(values)
    if model in incomplete or len(values) < 2:
        return None
    return round(statistics.stdev(values) / math.sqrt(len(values)), 3)


def _incomplete_models(log_dir: str, judge: str) -> list[str]:
    """Models whose newest run left a question unsettled, for the re-run loop.

    A stream error or an out-of-credit sample is worth another try, and costs nothing for the
    questions already cached. A policy refusal is not: the solver already tried three times, so it
    counts as settled here and the report fills it from a sibling instead.
    """
    n_items = len(load_items())
    answered: dict[str, int] = {}
    for log in _latest_logs(log_dir, judge):
        # Key by variant, not model. A skill variant shares the model name, so keying by model let
        # its count overwrite the bare model's and named a full model as short. Only bare ones come
        # back, because the caller re-runs each name as `--model` and cannot rebuild a skill one.
        variant = _variant(log)
        if variant != log.eval.model:
            continue
        answered[variant] = sum(
            1 for sample in log.samples or []
            if sample.scores and (sample.scores["rubric_judge"].value.get("answered")
                                  or sample.output.stop_reason == "content_filter")
        )
    return sorted(model for model, n in answered.items() if n < n_items)


def _rank(gt, table: list[dict], tol: dict[str, float] | None = None, **directions: str):
    """`ok_tables.cols_rank`, except a blank cell does not disqualify the whole column.

    cols_rank takes Python `min`/`max` over the column, which raises on a `None`, so one
    unanswered question stripped the arrow and the bold from every cell in that column.

    `tol` bolds everything within one standard error of the best, for the columns that carry
    one. A strict max bolded gpt-5.6-sol over gpt-5.6-terra on PC#7 by 0.0004, which a reader
    sees as two equal numbers with one of them singled out.
    """
    from great_tables import loc, style
    from ok_tables import ARROW, _label

    for column, direction in directions.items():
        values = [row[column] for row in table]
        real = [v for v in values if v is not None]
        if not real:
            continue
        best = min(real) if direction == "down" else max(real)
        near = (tol or {}).get(column, 0.0)
        rows = [i for i, v in enumerate(values) if v is not None and abs(v - best) <= near]
        gt = gt.cols_label(**{column: f"{_label(gt, column)}{ARROW[direction]}"})
        # A best that covers over half the column says nothing, so it is not bolded.
        if len(rows) * 2 <= len(real):
            gt = gt.tab_style(style.text(weight="bold"), loc.body(columns=column, rows=rows))
    return gt


def _incomplete_note(incomplete: dict[str, int], n_items: int) -> str:
    if not incomplete:
        return ""
    missing = ", ".join(f"{model.replace('openrouter/', '')} {n}/{n_items}"
                        for model, n in sorted(incomplete.items()))
    return (f"Answered fewer than {n_items} questions, so their aggregate columns are blank and "
            f"they are off the chart. Re-run them: {missing}\n")


def _latest_logs(log_dir: str, judge: str | None = None):
    """The fullest log per (model, judge), the key a score is defined on.

    Errored logs count, because a run that died partway still graded the samples it reached, and
    dropping it made the model vanish from the report instead of showing up as incomplete. Most
    answered samples wins, then newest, so a re-run that fails earlier than the first attempt does
    not throw away a complete result. Answered, not scored: a refusal is scored too, so counting
    scores kept a stale claude-fable-5 log with 9 answers over today's with 11.

    Directory layout is not load-bearing: a cross-judge run in a subdir is dropped by the judge
    column here, not by where its file sits. Calibration variants (mockllm/gold, ...) are not
    candidates under test and never pass the openrouter/ prefix.
    """
    import polars as pl

    logs = list(_graded_logs(log_dir, any_status=True))
    frame = pl.DataFrame({
        "i": range(len(logs)),
        "model": [_variant(log) for log in logs],
        "judge": [log.eval.metadata.get("judge") for log in logs],
        # Draws are kept side by side, not deduplicated: N answers to the same question are the
        # measurement, and keeping only the fullest would throw N-1 of them away.
        "draw": [_draw_of(log) for log in logs],
        "created": [log.eval.created for log in logs],
        "answered": [sum(1 for s in log.samples or []
                         if s.scores and s.scores["rubric_judge"].value.get("answered", 1.0))
                     for log in logs],
    }, schema={"i": pl.Int64, "model": pl.String, "judge": pl.String, "draw": pl.Int64,
               "created": pl.String, "answered": pl.Int64})
    if judge is not None:
        frame = frame.filter(pl.col("judge") == judge)
    frame = frame.filter(pl.col("model").str.starts_with("openrouter/"))
    keep = (frame.sort(["answered", "created"], descending=True)
                .unique(subset=["model", "judge", "draw"], keep="first"))["i"].to_list()
    return [logs[i] for i in keep]


def _borrow(rows: dict[str, dict[str, float]], refused: set[tuple[str, str]],
            n_items: int) -> dict[str, tuple[str, set[str]]]:
    """A row whose provider blocked a question reads a sibling's score there.

    Returns model -> (donor, the items the donor answered in its place). The donor is the same
    company's best complete model, so the pair is one lab and one row cannot borrow from a rival.
    Only a question the model was asked and refused: one it never reached stays blank, so a run
    that died partway reads as incomplete instead of being papered over.
    """
    complete = {model: sum(scores.values()) / n_items
                for model, scores in rows.items() if len(scores) == n_items}
    borrowed: dict[str, tuple[str, set[str]]] = {}
    for model, scores in rows.items():
        gaps = {item for name, item in refused if name == _short_variant(model)} - set(scores)
        company = _company(_model_of(model).replace("openrouter/", ""))
        donors = sorted((mean, donor) for donor, mean in complete.items()
                        if donor != model and gaps <= set(rows[donor])
                        and _company(_model_of(donor).replace("openrouter/", "")) == company)
        if gaps and donors:
            donor = donors[-1][1]
            scores.update({item: rows[donor][item] for item in gaps})
            borrowed[model] = (donor, gaps)
    return borrowed


def _results(log_dir: str, judge: str) -> None:
    """One row per (model, item), so a bad item shows up as a column every model fails.

    A sample the model never answered (empty, refused, provider-blocked) is blank, not 0.00: it
    says nothing about the model. `answered` counts how many of the items produced a gradable
    answer, and a model well below the item count is fighting the harness, not the questions.
    """
    import polars as pl
    from great_tables import GT, loc, md, style
    from ok_tables import cols_tip, to_markdown
    from tabulate import tabulate

    rows: dict[str, dict[str, float]] = {}
    # variant -> item -> one score per draw. `rows` holds the mean; this holds what it was made of.
    per_draw: dict[str, dict[str, list[float]]] = {}
    unanswered: list[dict] = []
    # Samples the judge could not grade at all. Not a refusal: the answer exists.
    ungraded: list[dict] = []
    # Cells the panel graded a seat short, because that seat errored. The cell still has a score.
    dropped: list[dict] = []
    # (short model name, item) the model was asked and did not answer, so the cell reads * rather
    # than blank. A cell missing from this set was never asked, e.g. the run is still going.
    refused: set[tuple[str, str]] = set()
    words: dict[str, list[int]] = {}
    # (answer key, seat, anchored score) for the seat-leniency fit, and which seats graded a model
    # at all, which is what the own-family dropout changes.
    judgments: list[tuple[str, str, float]] = []
    sat: dict[str, set[str]] = {}
    for log in _latest_logs(log_dir, judge):
        for sample in log.samples:
            variant = _variant(log)
            if not sample.scores:
                # The request itself failed, so there is no answer and no row. o1's two 404 runs
                # left a bare o1 row of 0/12 beside its real effort:low row, and the uplift table
                # then subtracted a None.
                if sample.error is not None:
                    continue
                # No score at all: the judge failed on an answer that exists, which is a fault in
                # this harness and not a fact about the model. Kept apart from a refusal, because
                # counting it as one told readers Anthropic blocks a question it never saw.
                rows.setdefault(variant, {})
                ungraded.append({
                    "model": _short_variant(variant),
                    "question": _SHORT_ITEM[sample.id][0],
                    "why": str(sample.error.message if sample.error else "no score")[:300],
                })
                continue
            score = sample.scores["rubric_judge"].value
            if score.get("answered", 1.0):
                # Every draw of this question, in one list. The cell is their mean, and their
                # spread is the part of the error bar that belongs to the model.
                per_draw.setdefault(variant, {}).setdefault(sample.id, []).append(score["score"])
                rows.setdefault(variant, {})[sample.id] = statistics.mean(per_draw[variant][sample.id])
                words.setdefault(variant, []).append(
                    len(sample.scores["rubric_judge"].answer.split()))
                meta = sample.scores["rubric_judge"].metadata
                # One row per grade, for the seat-leniency fit below. `pass_scores` is raw, so it
                # goes through the same anchor map the headline score used.
                for seat, raw in zip(meta.get("graded_by", []), meta["pass_scores"]):
                    seat = seat.removeprefix("openrouter/")
                    judgments.append((f"{variant}/{sample.id}", seat, _on_anchors(seat, raw)))
                    sat.setdefault(variant, set()).add(seat)
                for note in meta.get("failed_seats", []):
                    dropped.append({"model": _short_variant(variant),
                                    "question": _SHORT_ITEM[sample.id][0],
                                    "seat": note.split(":")[0].removeprefix("openrouter/"),
                                    "why": note.split(": ", 1)[-1][:200]})
            else:
                rows.setdefault(variant, {})
                refused.add((_short_variant(variant), sample.id))
                # Why, in the provider's words, so a refusal is attributable and not just a gap.
                # A blank cell we caused reads differently from one the provider caused, and both
                # arrive here as an empty answer. mimo-v2.5-pro wrote 40,000 tokens on VJ#12,
                # every one of them reasoning, and our own cap cut it off. Naming that stops a
                # reader counting it as a refusal. -- CLAUDE
                stop = getattr(sample.output, "stop_reason", None)
                why = sample.output.error or (
                    f"we truncated it: the answer hit ANSWER_MAX_TOKENS ({ANSWER_MAX_TOKENS:,}) "
                    f"with no text out" if stop == "max_tokens" else "empty answer")
                unanswered.append({
                    "model": _short_variant(variant),
                    "question": _SHORT_ITEM[sample.id][0],
                    # Which draw, or the table reads as duplicate rows: a question blocked in two
                    # draws of three is two rows here and still has a score in the main table.
                    "draw": _draw_of(log),
                    # Long enough for Anthropic's refusal, which ends in the policy link.
                    "why": why[:300],
                })
    # `answered` counts the model's own answers, taken before any fallback fills a refused cell.
    seat_effects = _seat_effects(judgments) if len(set(s for _, s, _ in judgments)) > 1 else {}
    # What a row gains or loses purely from who was in the room. The full panel averages to zero
    # by construction, so a row's shift is the mean leniency of the seats that did sit on it.
    dropouts = {model: statistics.mean(seat_effects[s] for s in seats)
                for model, seats in sat.items() if seat_effects}
    # What is left after the anchors and the leniency are taken out: how much two seats still
    # disagree about the same answer. This is the floor on the bench's resolution, not the
    # resolution: it holds the 12 questions fixed. The `+-` column adds the spread between them.
    spread = _judge_spread(judgments, seat_effects)
    # Judgments behind each row, for its own error bar.
    per_row: dict[str, int] = {}
    for answer, _, _ in judgments:
        per_row[answer.rsplit("/", 1)[0]] = per_row.get(answer.rsplit("/", 1)[0], 0) + 1
    own_answered = {model: len(scores) for model, scores in rows.items()}
    n_items = len(load_items())
    borrowed = _borrow(rows, refused, n_items)
    # A model graded on 10 of 12 questions is not comparable to one graded on 12, so its aggregate
    # columns are blank. The row and its per-item scores stay, because how it did on the questions
    # it did answer is still worth seeing.
    incomplete = {model: len(scores) for model, scores in rows.items() if len(scores) < n_items}
    items = sorted({item for scores in rows.values() for item in scores})
    # An exploratory or open item scores agreement with wassname, not correctness: its gold answer
    # is a current best guess, and for VG#11 the hypothesis came out null. Pooling
    # those into one mean would report agreement as if it were skill.
    settled = {i["id"] for i in load_items() if i["source"]["status"] in ("shipped", "validated")}
    usage = _usage(log_dir)
    table = [
        # Model name only. The company is the marker colour on the chart, and the vendor prefix
        # made every row label twice as long as it needed to be.
        {"model": _short_variant(model),
         # Hidden in the table, used for the chart colour and the hover card.
         "company": _company(_model_of(model).replace("openrouter/", "")),
         # Also hidden, also for the hover card: the row name says a fallback ran, this says which
         # model and how many questions.
         "fallback": _fallback_note(borrowed.get(model), n_items),
         "score": _mean_if_complete(scores.values(), model, incomplete),
         # Standard error of the row mean over the 12 questions. The judge-only error bar was
         # about 0.019 and invited a reading it cannot support: it says nothing about the luck of
         # which 12 questions were asked, which is four times larger and swings whole pairs.
         # One answer per question, so the sampling noise inside a cell is still not counted.
         "+-": _item_se(scores.values(), model, incomplete),
         "answered": f"{own_answered[model]}/{n_items}",
         # How many times the model answered each question. The cell is their mean, so a row with
         # more answers is the same measurement taken more carefully, not a different one. The
         # fewest, not the most: a sweep that ran out of budget partway through a draw leaves a row
         # with 2 answers to some questions and 1 to others, and the most would claim a precision
         # the weakest cell does not have.
         "answers/q": min((len(v) for v in per_draw.get(model, {}).values()), default=1),
         # The model's own tokens for the 12 questions. Not the judge's bill, which is bench
         # overhead, is about the same for every candidate, and hid the cheap end of the axis.
         "$/run": usage.get(model, {}).get("usd"),
         # Tokens per answer, reasoning included, so the cell compares directly against
         # ANSWER_MAX_TOKENS. It read as a per-answer mean while it was a 12-question sum, which made
         # a model at 1.3k per answer look like it had blown a 4k budget six times over.
         # Blank, not zero, when every answer came from cache: the cache records no usage, so the
         # number is unknown, and a 0.00 read as a real result got bolded as the best row.
         "tok/answer": _per_answer(usage.get(model, {}).get("out"), n_items),
         # Score per thousand tokens. A model that thinks three times as long for the same score is
         # worse to work with, and cost cannot show that: an expensive model can still be terse.
         "pts/Mtok": _ratio(_mean_if_complete(scores.values(), model, incomplete),
                            _millions(usage.get(model, {}).get("out"))),
         # Not a column, only the hover card. Every model is asked for the same effort, and this
         # says what each one did with it: 0.92 of deepseek-v4-flash-0731's output is thinking and
         # 0.03 of gemini-3.7-flash's, on the same setting.
         "reasoning share": _ratio(usage.get(model, {}).get("reasoning"),
                                   usage.get(model, {}).get("out")),
        **{item: scores.get(item) for item in items}}
        for model, scores in rows.items()
    ]
    # Blank scores sort last, so an incomplete model never sits in the middle of the ranking.
    table.sort(key=lambda row: (row["score"] is None, -(row["score"] or 0)))
    if not table:
        print(f"No graded logs at {RUBRIC_VERSION} in {log_dir}.")
        return
    # An variant answers the same 12 questions as its bare model, so it belongs beside that model in
    # the uplift table, not ranked against 25 other models where it reads as a 26th competitor.
    variants = [row for row in table if _VARIANT_LABEL.search(row["model"])]
    table = [row for row in table if row not in variants]
    # Row indices keyed on the short model name, taken before a fallback row is renamed.
    refused_idx = {item: [i for i, row in enumerate(table) if (row["model"], item) in refused]
                   for item in items}
    fallback_idx = {i: gaps for i, row in enumerate(table)
                    for model, (_, gaps) in borrowed.items()
                    if row["model"] == _short_variant(model)}
    for i in fallback_idx:
        table[i]["model"] += " (with fallback)"
    floors = [i for i, row in enumerate(table)
              if "gemma-3-4b" in row["model"] or "mockllm" in row["model"]]
    # No header and no source note on the table itself: those are sentences, so they live in
    # docs/page.md.j2 above and below `{{ tables.main }}` where wassname can edit them.
    gt = (
        GT(pl.DataFrame(table, nan_to_null=True), rowname_col="model")
        .fmt_number(columns=["score", *items], decimals=2, force_sign=True)
        .fmt_number(columns=["+-"], decimals=3, pattern="±{x}")
        .fmt_number(columns=["pts/Mtok"], decimals=0)
        .fmt_number(columns=["tok/answer"], decimals=0)
        # Significant figures, not decimals: at 2 decimals muse-spark-1.2 and claude-fable-5 both
        # rounded to $0.00 and looked free.
        .fmt_number(columns=["$/run"], n_sigfig=2, pattern="${x}")
        .sub_missing(missing_text="")
        .cols_hide(columns=["company", "reasoning share", "fallback"])
        .cols_label(**{item: _SHORT_ITEM[item][0] for item in items})
        .tab_style(style.text(style="italic"), loc.stub(rows=floors))
    )
    # A cell the model was asked and did not answer reads *, and points at the Blank cells table.
    # A cell it was never asked stays blank, so a run still in flight is not called a refusal.
    for item, marked in refused_idx.items():
        if marked:
            gt = gt.sub_missing(columns=[item], rows=marked, missing_text="*")
    # A borrowed cell is the fallback model's score, so it reads italic and is not this model's.
    for i, gaps in fallback_idx.items():
        gt = gt.tab_style(style.text(style="italic"),
                          loc.body(columns=sorted(gaps), rows=[i]))
    # cols_rank compares against None, so a column holding a blank cell cannot be ranked. That
    # column is the one worth reading, so it stays in the table without the arrow.
    directions = {"score": "up", "pts/Mtok": "up", "$/run": "down", "tok/answer": "down",
                  **{item: "up" for item in items}}
    # Bold the whole tie, not the winner of a coin flip. A row carries its own error bar, and a
    # single cell rests on the four or five seats that graded it.
    bars = [row["+-"] for row in table if row["+-"] is not None]
    cell_se = spread["sd"] / math.sqrt(spread["judgments_per_row"] / n_items) if spread else 0.0
    tol = {"score": statistics.median(bars) if bars else 0.0,
           **{item: cell_se for item in items}}
    gt = _rank(gt, table, tol, **directions)
    gt = cols_tip(
        gt,
        score=("items/", "the model's answer as a fraction of wassname's own, graded by the "
                         "judge. Above 1.00 means the judge preferred the model's answer"),
        # Domain only. The bench is private, and the id or title would let a reader work
        # backwards to the question.
        **{item: (f"items/{item}.md", _SHORT_ITEM[item][1]) for item in items},
        answered=("AGENTS.md", f"the model's own answers. Below {n_items} the mean is not "
                               "comparable, so the score stays blank, unless a fallback model "
                               "fills the refused questions"),
        **{"tok/answer": ("AGENTS.md", f"tokens the model generated for one answer, reasoning "
                                       f"included. Every model is asked for a "
                                       f"{ANSWER_MAX_TOKENS:,}-token thinking budget, and a row "
                                       f"far above it is a provider that does not count reasoning "
                                       f"against the cap")},
        **{"pts/Mtok": ("AGENTS.md", "score per million tokens. A model that writes more for "
                                     "the same score ranks lower")},
    )
    # Every table the page shows, as markdown. The words that go around them live in
    # docs/page.md.j2, which is the one file to edit. -- CLAUDE
    tables = {"main": to_markdown(gt)}
    lifts = _uplift(variants, table, items)
    if lifts:
        # Same typography as the main table: score is the ranked column, bold covers the tie within
        # one error bar. The change against the row above is the difference of two printed scores,
        # so only its bar is a column, at 3 decimals because the smallest change worth reading here
        # is 0.013 and 2 decimals would hide whether it beat that bar.
        bars = [row["+-"] for row in lifts if row["+-"] is not None]
        lift_gt = (
            GT(pl.DataFrame(lifts, nan_to_null=True), rowname_col="model")
            .fmt_number(columns=["score"], decimals=2, force_sign=True)
            .fmt_number(columns=["+-"], decimals=3, pattern="±{x}")
            .fmt_number(columns=["tok/answer"], decimals=0)
            .sub_missing(missing_text="")
        )
        tables["uplift"] = to_markdown(_rank(
            lift_gt, lifts, {"score": statistics.median(bars) if bars else 0.0}, score="up"))
    else:
        tables["uplift"] = ""
    # Three charts against numbers this bench did not produce: two public tests, and the calendar.
    suffix = _report_suffix(judge)
    index_png, index_fit = _versus(table, "index", suffix)
    hle_png, hle_fit = _versus(table, "hle", suffix)
    time_png, time_fit = _versus(table, "released", suffix)
    # Which row the line misses worst, and best, on each public number. Two of the three sections
    # are about that row, so it is measured once here and named in the template.
    for name, fit in (("index", index_fit), ("hle", hle_fit), ("released", time_fit)):
        low, high = min(zip(fit["resid"], fit["models"])), max(zip(fit["resid"], fit["models"]))
        fit["worst"] = {"model": low[1], "resid": low[0]}
        fit["best"] = {"model": high[1], "resid": high[0]}
        # A log fit's slope is a rate, so it reports a doubling time and not a score per year.
        fit["per_year"] = None if fit["log"] else fit["slope"] * 365.25
        fit["half"] = _doubling_months(fit) if fit["log"] else None
        fit["reach_month"] = f"{date.fromisoformat(fit['reach']):%B %Y}" if fit["reach"] else None
    below = index_fit["worst"]["model"]
    # The uplift table holds the one measurement that tests the effort story on the worst row, so
    # the two sections have to point at each other or a reader has to find it. -- CLAUDE
    lifted = [row for row in lifts if row["model"] == below and row["lift"] is not None
              and row["variant"].startswith("effort:")]
    effort_test = None
    if lifted:
        predicted = float(_fit_at(index_fit, PUBLIC_INDEX[below])[0])
        effort_test = {"variant": lifted[0]["variant"], "score": lifted[0]["score"],
                       "predicted": predicted, "after": lifted[0]["score"] - predicted}
    # What the other choices of fit would have said, so the caveats in the text are measured.
    alt = _timeline_alts(table)
    alt["first_when"] = f"{alt['first_when']:%b %Y}"
    alt["all_reach_month"] = f"{alt['all_reach']:%b %Y}" if alt["all_reach"] else None
    # Two different counts, and one sentence holding both read as a contradiction: "4 of 12 here.
    # The 1 it never answered". A refusal is stochastic, so with several draws a question can be
    # blocked in one draw and answered in another, and only a question blocked in every draw leaves
    # a gap to fill.
    notes = {"borrowed": [{"model": _short_variant(model),
                           "refused": sum(1 for name, _ in refused
                                          if name == _short_variant(model)),
                           "blocked": [_SHORT_ITEM[i][0] for i in sorted(gaps)],
                           "fallback": _short_variant(fallback)}
                          for model, (fallback, gaps) in sorted(borrowed.items())]}
    # Cost and tokens are measured, not quoted from a spec sheet, and a cached answer records no
    # usage. Say which rows are missing it and which are scaled, so a blank is not read as free.
    too_few = {m: usage[m]["questions"] for m in rows
               if usage.get(m, {}).get("out") is None and usage.get(m, {}).get("questions")}
    unpriced = sorted(_short_variant(m) for m in rows if usage.get(m, {}).get("out") is None
                      and m not in too_few)
    partial = sorted((_short_variant(m), usage[m]["questions"]) for m in rows
                     if usage.get(m, {}).get("out") is not None
                     and usage[m].get("questions", n_items) < n_items)
    notes["unpriced"] = unpriced
    notes["too_few"] = [{"model": _short_variant(m), "n": n} for m, n in sorted(too_few.items())]
    notes["partial"] = [{"model": m, "n": n} for m, n in partial]
    # Only a model that answered more than once can show what its bar is made of. One note, from
    # the model with the most draws: at v94 one model had repeats, at v96 every model does, and a
    # note per row put 24 near-identical paragraphs under the table.
    # The same model feeds the note, the charts and results.json, so the page and the text under it
    # cannot describe different rows. Most draws first, because that row's split is measured best,
    # then the median of those: the widest of them is a real row but reads as the typical bar.
    comps = _components(per_draw, spread, n_items)
    best = max((p["draws"] for p in comps.values()), default=0)
    pool = sorted(((k, v) for k, v in comps.items() if v["draws"] == best),
                  key=lambda kv: kv[1]["achieved"])
    twin, parts = pool[len(pool) // 2] if pool else (None, None)
    if parts:
        # The published parts, the same ones the chart draws, which are per row and not per draw.
        # Quoting the per-draw numbers here put 0.038 in the text against 0.022 in the chart legend,
        # a factor of root 3, under one label. -- CLAUDE
        split = _achieved_split(parts, n_items)
        notes["bar"] = {
            "model": _short_variant(twin), "draws": parts["draws"], "parts": split,
            "total": math.hypot(*split.values()),
            "single": parts["total"] / math.sqrt(n_items),
            "row_bar": next((r["+-"] for r in table + variants
                             if r["model"].startswith(_short_variant(twin))), None),
        }
    else:
        notes["bar"] = None
    # Two charts, same table: dollars are the vendor's price, tokens are what the model spends.
    cost_png = _pareto(table, "$/run", suffix, parts)
    token_png = _pareto(table, "tok/answer", suffix, parts)
    tables["unanswered"] = tabulate(
        sorted(unanswered, key=lambda r: (r["model"], r["question"], r["draw"])),
        headers="keys", tablefmt="pipe") if unanswered else ""
    tables["ungraded"] = tabulate(sorted(ungraded, key=lambda r: (r["model"], r["question"])),
                                  headers="keys", tablefmt="pipe") if ungraded else ""
    # A cell graded a seat short is our bookkeeping, not something the reader needs: the cell still
    # has a score, from the seats that remained. `failed_seats` in the log metadata is where to look
    # when a seat starts failing often enough to want replacing.
    # The panel and its calibration. A single judge has no seats to compare, so the whole section
    # belongs to a panel run, and an empty table is how the template knows.
    tables["judges"] = tabulate(
        [{"judge": seat, "judgments": sum(1 for _, s, _ in judgments if s == seat),
          "off-topic": JUDGE_ANCHORS[seat][0], "gold": JUDGE_ANCHORS[seat][1],
          "gap": JUDGE_ANCHORS[seat][1] - JUDGE_ANCHORS[seat][0], "leniency": leniency}
         for seat, leniency in sorted(seat_effects.items(), key=lambda kv: -kv[1])],
        headers="keys", tablefmt="pipe", floatfmt="+.3f") if seat_effects else ""
    # docs/index.html draws its own chart from this and drops the table in beside it. Same numbers
    # as the markdown and the pngs, one writer.
    full_id = {_short_variant(model): _model_of(model).replace("openrouter/", "")
               for model in rows}
    (Path(__file__).parent / "docs" / f"results{suffix}.json").write_text(json.dumps({
        "version": RUBRIC_VERSION,
        # Item ids do not travel. `_SHORT_ITEM` says why: an id describes its question closely
        # enough to work backwards from, and this file is served publicly, so the per-item cells go
        # out under the column label the table shows. -- CLAUDE
        # `index` and `released` are the two public x axes the page can draw against, so its own
        # chart covers all four axes and the pngs are only the no-javascript fallback.
        "rows": [{**{k: v for k, v in r.items() if k not in _SHORT_ITEM},
                  "full": full_id.get(r["model"].split(" (")[0], r["model"]),
                  "index": PUBLIC_INDEX.get(r["model"].split(" (")[0]),
                  "hle": AA_HLE.get(r["model"].split(" (")[0]),
                  "released": RELEASED.get(r["model"].split(" (")[0]),
                  **{_SHORT_ITEM[i][0]: r.get(i) for i in items}}
                 for r in table if r["score"] is not None],
        # The variants are out of `rows` so the chart plots one point per model. What a skill or a
        # raised effort bought that model lives here instead, paired by question.
        "uplift": lifts,
        # The two fits drawn in sandbagging.png and timeline.png, so the page can quote them without
        # refitting: score against a public index, and score against release date. `resid` is in
        # `models` order, and a row far below the line is the sandbagging one.
        "fits": {"index": index_fit, "hle": hle_fit, "released": time_fit},
        # Keyed on the column label, not the item id, and the domain is the only other thing about a
        # question this bench publishes.
        "items": {_SHORT_ITEM[item][0]: {"domain": _SHORT_ITEM[item][1]} for item in items},
        "logos": {c: f"logos/{f}" for c, f in _COMPANY_LOGOS.items()},
        # Header tooltips for the page, in plain english. The source note under the table says the
        # same things at length; a reader who hovers a column wants one sentence.
        "columns": _COLUMN_HELP,
        # Who graded, how each seat is calibrated, and what the own-family dropout costs. A list
        # either way, so a single judge and a panel read the same on the page.
        # The default for the table: every model at the quietest rung it lists, which is not one
        # level and is not each model's own default, so the page has to say so. A row run at a
        # named effort carries it in its own name, `grok-4.6 (effort:high)`.
        "reasoning_effort": EFFORT_ARM,
        "judge": {"models": judge.replace("openrouter/", "").split("+"),
                  "passes": JUDGE_PASSES, "temperature": JUDGE_TEMPERATURE,
                  # Each seat's two measured anchors, from `just calibrate`: what it scores an
                  # unrelated gold answer and what it scores this item's own gold answer. Every
                  # score it gives is read through them, so 0 and 1 mean the same on every seat.
                  # `leniency` is what it adds in the middle, where the anchors do not reach.
                  "seats": {seat: {"off_topic": JUDGE_ANCHORS[seat][0],
                                   "gold": JUDGE_ANCHORS[seat][1],
                                   "gap": round(JUDGE_ANCHORS[seat][1] - JUDGE_ANCHORS[seat][0], 3),
                                   "leniency": round(leniency, 3),
                                   "judgments": sum(1 for _, s, _ in judgments if s == seat)}
                            for seat, leniency in sorted(seat_effects.items())},
                  # What two seats still disagree by on one answer once the anchors and the
                  # leniency are out, and what that leaves on a whole row. This is one term of
                  # `uncertainty` below, and the smallest of the three, so a page that shows it
                  # alone tells the reader the bench is four times sharper than it is.
                  "disagreement": spread,
                  # A seat never grades its own company, so those rows are graded by a different
                  # mixture. This is what that costs them, in score, against a full panel.
                  "dropout": {_short_variant(model): round(dropout, 3)
                              for model, dropout in sorted(dropouts.items())
                              if round(dropout, 3)}},
        # What one row's error bar is made of, so the page can draw the same split the charts do.
        # Null when no model answered twice, since there is then nothing to separate the answer
        # term from the question term. Measured on `draws` answers by `model`.
        # Row units, the same units as the `+-` column, so the page can draw this beside a model.
        # `_components` works per question, and publishing that raw put 0.32 next to a row bar of
        # 0.09. `achieved` is the one that already holds the draw averaging, so it is the total,
        # and the parts are the split that adds to it.
        "uncertainty": parts and {
            "model": _short_variant(twin),
            "draws": parts["draws"],
            "total": round(parts["achieved"] / math.sqrt(n_items), 3),
            "parts": {name: round(sd, 3) for name, sd in _achieved_split(parts, n_items).items()},
        },
        # A blank cell and the provider's own words for it. The page marks these * and does not
        # read them as zero.
        "unanswered": sorted(unanswered, key=lambda r: (r["model"], r["question"])),
        # A row that reads a sibling's score on the questions its provider blocked. The page marks
        # these cells italic and names the pair.
        "borrowed": {_short_variant(model): {"from": _short_variant(fallback),
                                         "items": sorted(_SHORT_ITEM[i][0] for i in gaps),
                                         "refused": sum(1 for name, _ in refused
                                                        if name == _short_variant(model))}
                     for model, (fallback, gaps) in borrowed.items()},
    }, indent=1))
    # Two sections a reader of the public page never sees: which questions separate the models, and
    # what the bench itself cost. Captured whole, because the template only places them.
    tables["per_item"] = _capture(_per_item, rows, settled)
    tables["tokens"] = _capture(_tokens, log_dir)
    ctx = {"tables": tables, "notes": notes, "alt": alt, "effort_test": effort_test,
           "fits": {"index": index_fit, "hle": hle_fit, "released": time_fit},
           "charts": {"index": index_png.name, "hle": hle_png.name, "released": time_png.name,
                      # Bare file names, because this markdown sits in docs/ beside the charts. A
                      # path from the repo root broke the image on GitHub.
                      "cost": cost_png.name, "tokens": token_png.name},
           "n_items": n_items, "version": RUBRIC_VERSION,
           "best_score": max(r["score"] for r in table if r["score"] is not None),
           "effort": EFFORT_ARM,
           "effort_gap_max": max(AA_EFFORT_GAP.values())}
    print(_render_page(ctx, private=True), end="")
    # The public copy is the same template with the private sections switched off, so nothing has to
    # strip item links back out of finished markdown.
    (Path(__file__).parent / "docs" / f"results{suffix}_public.md").write_text(
        _render_page(ctx, private=False))


def _capture(fn, *args) -> str:
    """Run a function that prints, and keep what it printed."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        fn(*args)
    return out.getvalue().strip()


def _render_page(ctx: dict, private: bool) -> str:
    """docs/page.md.j2 holds every word of the report, and this puts the numbers in it.

    `StrictUndefined`, so a sentence that quotes a name the data does not have fails the build
    instead of printing an empty space nobody notices.
    """
    from jinja2 import Environment, FileSystemLoader, StrictUndefined

    env = Environment(loader=FileSystemLoader(Path(__file__).parent / "docs"),
                      undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)
    env.filters |= {"s2": lambda v: f"{v:+.2f}", "s3": lambda v: f"{v:+.3f}",
                    "d2": lambda v: f"{v:.2f}", "d3": lambda v: f"{v:.3f}",
                    "n0": lambda v: f"{v:.0f}"}
    # One blank line between blocks, never three: a `{% if %}` that skips leaves its own newline.
    page = env.get_template("page.md.j2").render(private=private, **ctx)
    if not private:
        # The table is generated, so its item links are stripped here rather than in the template.
        # `[AP#1↑][<item id>]` keeps the label and loses the link, and the definition that
        # named the item file goes with it.
        page = re.sub(r"\[([^\]]+)\]\[[^\]]+\]", r"\1", page)
        page = re.sub(r"(?m)^\[[^\]]+\]: (items/|\S+ \").*\n?", "", page)
        # Both pareto pngs: the page draws that axis itself, at the top and under its own chip row.
        page = re.sub(r"(?m)^!\[[^\]]*\]\(pareto[^)]*\)\n?", "", page)
        assert "items/" not in page, "an item link survived"
    return re.sub(r"\n{3,}", "\n\n", page).strip() + "\n"


# One sentence per column, for the page's header tooltips. Plain english, because a reader hovers
# a column when the label alone did not tell them enough.
_COLUMN_HELP = {
    "model": "The model that answered the questions.",
    "score": "The mean score of all 12 answers. 1.00 is wassname's own answer to the question."
             " Above 1.00 means the judge preferred the model's answer.",
    "+-": "Standard error of the row over the 12 questions. It covers both which questions were "
          "asked and how the seats read them. Two rows whose bars overlap are not separated.",
    "answered": "How many of the 12 questions the model answered. Below 12 the mean is not "
                "comparable, so the score stays blank.",
    "answers/q": "Each score is the mean of this many answers to the same question, counted on the "
                 "question that got the fewest. A row that answered every question three times "
                 "reads 3; a row that refused one question twice reads 1, even though the other "
                 "eleven have three answers. More answers is the same measurement taken more "
                 "carefully, not an easier question.",
    "$/run": "The cost of one run of all 12 questions. This counts the model's own tokens, not "
             "the judge's.",
    "tok/answer": "Tokens the model generated for one answer, reasoning included.",
    "pts/Mtok": "Score per million tokens. A model that writes more for the same score ranks "
                "lower.",
}
# Short column labels: initials plus the question number. The item id is several times the width
# of the number under it, which doubled the table. The tooltip carries the id and the neutral
# title. Not the description, which would help someone benchmax the answer.
# The bench stays private, so the tooltip names the domain and nothing else. The item id and title
# both describe the question closely enough to work backwards from.
# The map itself lives in items/, which does not travel, because this file does.
_SHORT_ITEM = {k: v for k, v in
               json.loads((Path(__file__).parent / "items" / "_short.json").read_text()).items()
               if not k.startswith("_")}


def _company(model: str) -> str:
    """Company label for chart coloring, from the model's vendor prefix."""
    return {
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "deepseek": "DeepSeek",
        "google": "Google",
        "x-ai": "xAI",
        "qwen": "Alibaba",
        "moonshotai": "Moonshot AI",
        "z-ai": "Zhipu",
        "meta": "Meta",
        "minimax": "MiniMax",
        "xiaomi": "Xiaomi",
        "thinkingmachines": "Thinking Machines",
        # OpenRouter's cloak for a model whose lab is not announced yet.
        "stealth": "Stealth",
    }[model.split("/")[0]]


# Which logo the page draws for each lab. The files live in the public repo, github.com/wassname/
# ml-bench, from simpleicons.org, or the lab's own icon for the four simpleicons has no mark for.
# The png below wears plain rings, because it is read beside the model names in results.md.
_COMPANY_LOGOS = {
    "OpenAI": "openai.png",
    "Anthropic": "anthropic.svg",
    "DeepSeek": "deepseek.svg",
    "Google": "google.svg",
    "xAI": "x-ai.svg",
    "Alibaba": "qwen.svg",
    "Moonshot AI": "moonshotai.svg",
    "Zhipu": "z-ai.png",
    "Meta": "meta.svg",
    "MiniMax": "minimax.svg",
    "Xiaomi": "xiaomi.svg",
    "Thinking Machines": "thinkingmachines.png",
}
_RING_PX = 9
_RING_INK = "#6b7280"
_LABEL_INK = "#1f2937"

# Two public numbers per model, neither of them ours, so a score can be read against something this
# bench did not produce. `PUBLIC_INDEX` is the Artificial Analysis intelligence index headline, and
# `AA_EFFORT_GAP` is what AA says the same model loses at its lowest reasoning setting, which is the
# confound in any residual: this bench runs at `minimal`, AA headlines mostly at max.
# Read 2026-08-16 from https://artificialanalysis.ai/leaderboards/models, with the two spot checks
# in docs/slop/ref/aa_index.md. Not listed there, so out of the index fit: qwen3.7-flash,
# qwen3.8-27b, both AA pages 404.
PUBLIC_INDEX = {
    "claude-opus-5": 63, "claude-fable-5": 62, "gpt-5.6-sol": 61, "grok-4.6": 61, "kimi-k3": 60,
    "qwen3.8-max": 58, "gpt-5.6-terra": 57, "muse-spark-1.2": 57, "gemini-3.7-flash": 56,
    "claude-sonnet-5": 55, "glm-5.2": 53, "deepseek-v4-pro-0813": 53, "gpt-5.6-luna": 52,
    "deepseek-v4-flash-0731": 52, "gemini-3.6-flash": 52, "minimax-m3": 45, "mimo-v2.5-pro": 43,
    "inkling": 42, "inkling-small": 41, "claude-haiku-4.5": 30, "gemma-4-31b-it": 30,
    "gpt-oss-120b": 24, "qwen3.5-9b": 22,
    # The six older flagships, read 2026-08-17 from the same model pages, top setting each:
    # Opus 4.8 max effort, GPT-5.5 xhigh, Grok 4.5 high, GLM-5.1 and the two 27Bs reasoning.
    "claude-opus-4.8": 57, "gpt-5.5": 56, "grok-4.5": 56, "glm-5.1": 41, "qwen3.6-27b": 38,
    "qwen3.5-27b": 35,
}
AA_EFFORT_GAP = {"claude-opus-5": 11, "gpt-5.6-sol": 10, "kimi-k3": 12, "gpt-5.6-terra": 16,
                 "gemini-3.7-flash": 5, "gpt-5.6-luna": 18, "gpt-oss-120b": 9, "gpt-5.5": 12}
# Humanity's Last Exam, percent, from the same AA model pages, read 2026-08-17. AA re-runs HLE with
# its own grader, so these are not the Scale SEAL numbers: muse-spark-1.2 is 45.5 here, 40.56 there.
# All 23 rows are the model's top setting, like the index above, and each one's `intelligenceIndex`
# in the same record rounds to its `PUBLIC_INDEX` value, which is how the setting was checked.
# Per-row sources in docs/slop/ref/aa_hle.md. claude-sonnet-5's low, medium, high and xhigh pages all
# carry hle null, so only its top row is scored.
AA_HLE = {
    "claude-fable-5": 55.5, "claude-opus-5": 54.9, "gpt-5.6-sol": 49.5, "gemini-3.7-flash": 47.9,
    "kimi-k3": 46.9, "muse-spark-1.2": 45.5, "qwen3.8-max": 43.0, "grok-4.6": 42.9,
    "gpt-5.6-terra": 42.9, "claude-sonnet-5": 41.3, "glm-5.2": 41.1, "deepseek-v4-pro-0813": 41.0,
    "gemini-3.6-flash": 40.8, "gpt-5.6-luna": 39.5, "minimax-m3": 39.0,
    "deepseek-v4-flash-0731": 38.6, "mimo-v2.5-pro": 35.7, "inkling-small": 33.3, "inkling": 31.9,
    "gemma-4-31b-it": 23.6, "gpt-oss-120b": 19.6, "qwen3.5-9b": 14.9, "claude-haiku-4.5": 10.4,
    "claude-opus-4.8": 48.7, "gpt-5.5": 45.8, "grok-4.5": 42.7, "glm-5.1": 30.1,
    "qwen3.5-27b": 23.9, "qwen3.6-27b": 23.1,
}
# Release date per model, from the same site's model pages, read 2026-08-17. Two models AA does not
# list carry their OpenRouter listing date instead, which is an upper bound on the announcement.
# The per-row sources and the four AA/OpenRouter disagreements are in docs/slop/ref/release_dates.md.
RELEASED = {
    "gpt-oss-120b": "2025-08-05", "claude-haiku-4.5": "2025-10-15", "qwen3.5-9b": "2026-03-02",
    "gemma-4-31b-it": "2026-04-02", "mimo-v2.5-pro": "2026-04-22", "minimax-m3": "2026-06-01",
    "claude-fable-5": "2026-06-09", "glm-5.2": "2026-06-16", "claude-sonnet-5": "2026-06-30",
    "gpt-5.6-terra": "2026-07-09", "gpt-5.6-sol": "2026-07-09", "gpt-5.6-luna": "2026-07-09",
    "inkling": "2026-07-15", "kimi-k3": "2026-07-16", "gemini-3.6-flash": "2026-07-21",
    "claude-opus-5": "2026-07-24", "qwen3.7-flash": "2026-07-27", "inkling-small": "2026-07-30",
    "deepseek-v4-flash-0731": "2026-07-31", "qwen3.8-max": "2026-08-03",
    "muse-spark-1.2": "2026-08-05", "grok-4.6": "2026-08-12", "gemini-3.7-flash": "2026-08-13",
    "deepseek-v4-pro-0813": "2026-08-13", "qwen3.8-27b": "2026-08-14",
    # OpenRouter's own created field, 1787086655.
    "glm-5.3": "2026-08-18",
    # Same, for the two AA does not list: 1787256295 and 1787311563.
    "ox-alpha": "2026-08-20", "deepseek-v4-flash-vision-exp": "2026-08-21",
    "qwen3.5-27b": "2026-02-24", "glm-5.1": "2026-04-07", "qwen3.6-27b": "2026-04-22",
    "gpt-5.5": "2026-04-23", "claude-opus-4.8": "2026-05-28", "grok-4.5": "2026-07-08",
    # OpenAI's own announcement days, not AA, which lists neither.
    "gpt-4": "2023-03-14", "o1": "2024-12-05",
}


def _fit(x, y, log: bool = False) -> dict:
    """Least squares line, kept as parameters so the band and the residuals come from one fit.

    With `log` the fit is on log10 of the score, so the line is an exponential in score units. Its
    slope is a doubling time, and its `sd` is in log units, not score units.
    """
    import numpy as np

    x, y = np.asarray(x, float), np.asarray(y, float)
    if log:
        y = np.log10(y)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    return {"slope": float(slope), "intercept": float(intercept), "log": log,
            # Two parameters fitted, so n-2 in the denominator.
            "sd": float(resid.std(ddof=2)), "r": float(np.corrcoef(x, y)[0, 1]),
            "n": len(x), "mean": float(x.mean()), "sxx": float(((x - x.mean()) ** 2).sum()),
            "resid": [float(v) for v in resid]}


def _fit_at(fit: dict, q):
    """The line at q, and the standard error of the line there.

    s * sqrt(1/n + (q - xbar)^2 / Sxx), which widens as q leaves the data. This is how well the
    line is pinned, not where one more model would land: that interval is wider by the residual sd.
    """
    import numpy as np

    q = np.asarray(q, float)
    return (fit["slope"] * q + fit["intercept"],
            fit["sd"] * np.sqrt(1 / fit["n"] + (q - fit["mean"]) ** 2 / fit["sxx"]))


def _curve(fit: dict, q):
    """The line and its band at q, in score units, whichever space the fit was made in.

    A log fit lives in log10 of the score, so the line and its band come back through 10**.
    """
    line, se = _fit_at(fit, q)
    if fit["log"]:
        return 10 ** line, 10 ** (line - se), 10 ** (line + se)
    return line, line - se, line + se


def _reach(fit: dict, target: float = 1.0):
    """Where the line reaches `target`, in x units."""
    if fit["log"]:
        return (math.log10(target) - fit["intercept"]) / fit["slope"] if fit["slope"] > 0 else None
    return (target - fit["intercept"]) / fit["slope"] if fit["slope"] > 0 else None


def _doubling_months(fit: dict) -> float:
    """Months for the score to double, which is what a log slope means."""
    return 12 * math.log10(2) / (fit["slope"] * 365.25)


# The models this chart fits, chosen by wassname. A running best computed from the table is not
# the same thing: a weak model released in a quiet month sets a record here and joins the fit, which
# put gemma-4-31b-it and glm-5.1 on the frontier and pulled the line off the flagships.
FRONTIER = ["gpt-4", "o1", "gpt-5.5", "claude-fable-5"]


def _running_best(table: list[dict]) -> list[tuple[int, float, str]]:
    """wassname's frontier models, in release order, dropping any that has no score yet."""
    return sorted((date.fromisoformat(RELEASED[name]).toordinal(), row["score"], name)
                  for row in table
                  for name in [row["model"].split(" (")[0]]
                  if row["score"] is not None and name in FRONTIER)


def _frontier(table: list[dict], x: str) -> set[str]:
    """Models that nothing else beats on both axes: less x AND more score.

    The point of the bench is to disagree with a public leaderboard about which model to work
    with, and that decision is score against what a run costs you, in dollars or in tokens.
    """
    best = -2.0
    front = set()
    for row in sorted(table, key=lambda r: (r[x], -r["score"])):
        if row["score"] > best:
            front.add(row["model"])
            best = row["score"]
    return front


# Two charts, same data, two definitions of cheap. Dollars are a vendor's price, which changes
# without the model changing; tokens are what the model actually spends to answer, which is what
# you wait for and what a self-hosted run pays.
_X_AXES = {
    "$/run": {"stem": "pareto", "prefix": "$", "corner": "powerful and cheap",
              "subtitle": "Lower cost is better", "title": "$ per run, all {n} items"},
    "tok/answer": {"stem": "pareto_tokens", "prefix": "", "corner": "powerful and terse",
                   "subtitle": "Fewer tokens is better",
                   "title": "tokens per answer, reasoning included"},
}


def _report_suffix(judge: str) -> str:
    """Artifacts are named after the judge, so one judge's table cannot overwrite another's.

    The default judge writes the plain names, `results.md` and `pareto.png`. Any other judge writes
    `results_<judge>.md` and `pareto_<judge>.png`. Reading the deepseek table used to replace the
    chart the default judge had written, with nothing in the file saying who graded it.
    """
    name = judge.split("/")[-1]
    return "" if judge == JUDGE_MODEL else f"_{name}"


def _achieved_split(parts: dict, n_items: int) -> dict[str, float]:
    """The three parts in row units, after the draws are averaged.

    `_components` reports what one draw carries. A row that averages k draws keeps only a 1/k share
    of the answer term, so a chart that draws the per-draw parts beside a row's `+-` draws segments
    that overflow it.

    SHOULD: hypot of these equals `achieved / sqrt(n_items)`, the `+-` in the row. ELSE the bar and
    its own parts describe different rows.
    """
    return {name: sd / math.sqrt(n_items * (parts["draws"] if name == "between answers" else 1))
            for name, sd in parts["sd"]}


def _uncertainty_bar(fig, parts: dict, plotted: list[dict], x: str, x_range: tuple) -> None:
    """One representative error bar, cut into the three things that move a score.

    Not a bar per point: every row's bar is about the same size, so 25 of them would be ink for
    one number. A single typical bar in the corner is the convention, and the split says which
    part more draws can shrink (between answers) and which only more questions can (between
    questions).
    """
    n_items = len(load_items())
    ink = {"between questions": "#6b7280", "between answers": "#b45309", "between judges": "#c7cdd6"}
    row = _achieved_split(parts, n_items)
    total = math.hypot(*row.values())
    # Segment heights are variance shares, scaled so the whole bar is one standard error. Stacking
    # the standard errors themselves would draw a bar taller than the bar it describes, since
    # variances add and roots do not.
    shares = {name: total * (sd ** 2 / total ** 2) for name, sd in row.items()}
    # The empty low-score end of the cost axis, where no model sits.
    at = 10 ** (x_range[1] - 0.55)
    base = min(r["score"] for r in plotted) - 0.03
    y = base
    for name, height in shares.items():
        # A line trace, not a shape: traces take the value and the log axis converts it, while a
        # shape wants log10 of the value and drew nothing when handed either.
        fig.add_scatter(x=[at, at], y=[y, y + height], mode="lines", showlegend=False,
                        hoverinfo="skip", line={"color": ink[name], "width": 9})
        y += height
    fig.add_annotation(
        text=(f"±{total:.3f} uncertainty<br>"
              + "<br>".join(f"<span style='color:{ink[n]}'>&#9632;</span> {n} {row[n]:.3f}"
                            for n in shares)),
        x=math.log10(at) + 0.09, y=base, xanchor="left", yanchor="bottom",
        showarrow=False, align="left", font={"size": 10, "color": _LABEL_INK},
    )


def _pareto(table: list[dict], x: str = "$/run", suffix: str = "",
            parts: dict | None = None) -> Path:
    """The chart as a png, for results.md. The page draws its own from docs/results.json."""
    import plotly.graph_objects as go

    from place_labels import place_labels

    docs = Path(__file__).parent / "docs"
    axis = _X_AXES[x]
    # An incomplete run has no comparable mean, and a fully cached run recorded neither cost nor
    # tokens, so neither has a place on the chart.
    plotted = [r for r in table if r["score"] is not None and r[x]]
    assert plotted, (
        f"no model has a {x}, so there is nothing to plot. Every answer in this log directory "
        "replayed from the cache, and a cached answer records no tokens, so the count only exists "
        "in the log of the run that first paid for it. Report from the tree that holds those logs.")
    front = _frontier(plotted, x)
    fig = go.Figure()
    # Every bound comes from the data. 1.00 is wassname's own answer, so the axis always reaches
    # it even when nobody does, which is the gap the chart exists to show.
    top = max(1.05, max(r["score"] for r in plotted) + 0.08)
    # The floor follows the data. Pinning it at 0.0 spent half the canvas on empty space when the
    # weakest model scored 0.38, and 0.0 keeps its meaning in the caption rather than on the axis.
    floor = min(r["score"] for r in plotted) - 0.09
    spend = [r[x] for r in plotted]
    x_range = (math.log10(min(spend) / 3), math.log10(max(spend) * 3))
    size = {"fig_w": 1100, "fig_h": 680, "margin": {"l": 80, "r": 150, "t": 130, "b": 70}}
    # A dot and its name. Which lab it belongs to is the page's job, where a logo can be drawn.
    fig.add_scatter(
        x=[r[x] for r in plotted], y=[r["score"] for r in plotted],
        mode="markers", showlegend=False,
        marker={"color": "white", "size": _RING_PX, "line": {"color": _RING_INK, "width": 2}},
        # Pre-format the tooltip in Python: plotly's %{y:.2f} hovertemplate specifier is
        # unreliable across versions and silently falls back to full precision.
        hovertext=[
            f"{r['model']} ({EFFORT_ARM})<br>{r['company']}"
            f"<br>score {r['score']:+.2f}"
            + (f"<br>{r['fallback']}" if r.get("fallback") else "")
            + f"<br>${r['$/run']:.3g} per run"
              f"<br>{_ktok_text(r)}"
            for r in plotted
        ],
        hovertemplate="%{hovertext}<extra></extra>",
    )
    steps = sorted((r for r in plotted if r["model"] in front), key=lambda r: r[x])
    fig.add_scatter(x=[r[x] for r in steps], y=[r["score"] for r in steps],
                    mode="lines", line={"dash": "dot", "color": "#4b5563"},
                    name="Pareto line", hoverinfo="skip", showlegend=False)
    fig.add_hline(y=1.0, line={"dash": "dash", "width": 1, "color": "#9ca3af"},
                  annotation_text="1.00 = wassname's own answer",
                  annotation_position="top left",
                  annotation_font={"size": 10, "color": "#6b7280"})
    fig.update_layout(
        title=(
            "wassname-ml-bench<br>"
            f'<sub>{len(load_items())} problems from the research of '
            '<a href="https://wassname.org">wassname</a>. '
            f'1.00 is his own answer. {axis["subtitle"]}</sub>'
        ),
        # One labelled tick per decade. Plotly's default log axis labels every minor tick, which
        # put 2 3 4 5 6 7 8 9 between each decade and buried the two numbers that matter.
        xaxis={"title": axis["title"].format(n=len(load_items())), "type": "log",
               "range": list(x_range), "dtick": 1, "tickprefix": axis["prefix"],
               "minor": {"ticks": "outside", "showgrid": False}},
        yaxis={"title": "mean score", "zeroline": True, "range": [floor, top]},
        template="simple_white", width=size["fig_w"], height=size["fig_h"],
        margin=size["margin"],
    )
    # Labels collide badly at this density, and plotly has no native de-overlap
    # (plotly.js #4674). place_labels is wassname's greedy placer, vendored beside this file.
    # add_annotation one at a time, never update_layout(annotations=[...]): plotly merges that list
    # into the existing annotations by index, and add_hline had already put one at index 0, which
    # silently ate the first model's label.
    for label in place_labels(
        [{"x": math.log10(r[x]), "y": r["score"], "text": r["model"],
          "color": _LABEL_INK}
         for r in sorted(plotted, key=lambda r: -r["score"])],
        x_range=x_range, y_range=(floor, top),
        # Every marker is an obstacle, including a label's own. Without this the placer only
        # avoided other labels, so mimo-v2.5-pro's label sat on top of its own dot.
        obstacles=[(math.log10(r[x]), r["score"]) for r in plotted],
        **size,
    ):
        fig.add_annotation(**label)
    # An arrow, not a shaded quadrant: same message, a fraction of the ink, and no legend entry.
    fig.add_annotation(
        text=axis["corner"], xref="paper", yref="paper", x=0.02, y=0.94,
        ax=42, ay=34, axref="pixel", ayref="pixel", xanchor="left", yanchor="middle",
        showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor="#16a34a",
        font={"size": 12, "color": "#16a34a"},
    )
    fig.add_annotation(
        text=RUBRIC_VERSION, xref="paper", yref="paper", x=0.99, y=0.02,
        xanchor="right", yanchor="bottom", showarrow=False,
        font={"size": 10, "color": "#9ca3af"},
    )
    # Where the chart came from, so a copied png can still be traced back.
    fig.add_annotation(
        text=SITE_URL, xref="paper", yref="paper", x=0.0, y=0.02,
        xanchor="left", yanchor="bottom", showarrow=False,
        font={"size": 10, "color": "#9ca3af"},
    )
    if parts:
        _uncertainty_bar(fig, parts, plotted, x, x_range)
    png = docs / f"{axis['stem']}{suffix}.png"
    fig.write_image(png, scale=2)
    return png


# The other two charts: score against a number this bench did not produce. Same shape both times,
# a straight line with the standard error of the line drawn around it, so the two share one drawer.
# These three charts have a live twin in the public repo's index.html, `AXES` there. A png gets a
# title and a subtitle block, the svg gets an axis label and one note, so the wording differs; a
# claim changed here has to be changed there.
_VS_AXES = {
    "index": {"stem": "sandbagging", "x_unit": "index point",
              "x_title": "Artificial Analysis intelligence index",
              "title": "Private research problems against public tests",
              "subtitle": "A model below the line does worse on wassname's own research than "
                          "its public score predicts"},
    # A second public number, so the sandbagging read does not rest on one leaderboard column.
    "hle": {"stem": "hle", "x_unit": "point of the exam",
            "x_title": "Humanity's Last Exam, a STEM exam, percent, AA's own run",
            "title": "Private research problems against a public STEM exam",
            "subtitle": "A model below the line does worse on wassname's own research than "
                        "its exam score predicts"},
    "released": {"stem": "timeline", "x_title": "release date",
                 "title": "How long until a model reaches wassname's own answer?",
                 # The caveat rides the chart, not only the page: a png gets screenshotted alone,
                 # and this one names a year, which reads as a forecast unless the chart says no.
                 "subtitle": "An exponential through the frontier flagships, extended to 1.00."
                             "<br>Fitted as a straight line on log "
                             "score. The band is one standard error on that line, and these models "
                             "were picked to fill a table, not sampled over time"},
}


def _timeline_alts(table: list[dict]) -> dict:
    """The published fit is an exponential through the frontier. This is what the other two choices
    would say: every model rather than the frontier, and a straight line in score rather than in
    log score. Measured rather than described, so the page's caveats cannot go stale.
    """
    front = _running_best(table)
    every = [(date.fromisoformat(RELEASED[name]).toordinal(), row["score"])
             for row in table
             for name in [row["model"].split(" (")[0]]
             if row["score"] is not None and name in RELEASED]
    all_lin = _fit([p[0] for p in every], [p[1] for p in every])
    all_reach = _reach(all_lin, 1.0)
    return {"n_models": len(every), "all_per_year": all_lin["slope"] * 365.25,
            "all_sd": all_lin["sd"], "all_r": all_lin["r"],
            "all_reach": date.fromordinal(int(all_reach)) if all_reach else None,
            # The doubling time the published fit implies, and the one every model implies.
            "half_front": _doubling_months(_fit([p[0] for p in front], [p[1] for p in front],
                                                log=True)),
            "half_all": _doubling_months(_fit([p[0] for p in every], [p[1] for p in every],
                                              log=True)),
            "first": front[0][2], "first_score": front[0][1],
            "first_when": date.fromordinal(front[0][0]),
            "cheap": sum(1 for _, score, _ in front if score < 0.60)}


def _versus(table: list[dict], kind: str, suffix: str = "") -> tuple[Path, dict]:
    """Score against a public number, with the fit and the uncertainty of the fit.

    `index` answers the sandbagging question: which models do worse here than their public score
    says. `released` answers wassname's: when does the line reach 1.00, his own answer.
    """
    import numpy as np
    import plotly.graph_objects as go

    from place_labels import place_labels

    axis = _VS_AXES[kind]
    dated = kind == "released"
    known = {"released": RELEASED, "index": PUBLIC_INDEX, "hle": AA_HLE}[kind]
    # The fallback suffix is not part of the model's name on either public page.
    plotted = [{**r, "x": (date.fromisoformat(known[r["model"].split(" (")[0]]).toordinal()
                           if dated else known[r["model"].split(" (")[0]])}
               for r in table
               if r["score"] is not None and r["model"].split(" (")[0] in known]
    # The date chart fits the frontier, which wassname asked for: only the models that beat every
    # earlier one. Every model is still drawn, and only that subset is fitted. The fit is on log
    # score, so the line is an exponential and still reaches 1.00 on a date.
    front = _running_best(table) if dated else []
    fit = (_fit([p[0] for p in front], [p[1] for p in front], log=True) if dated
           else _fit([r["x"] for r in plotted], [r["score"] for r in plotted]))
    fit["models"] = [p[2] for p in front] if dated else [r["model"] for r in plotted]
    # Where the line reaches 1.00. Only on the date axis: an index point is not a thing to wait for,
    # and feeding index 89 to `date.fromordinal` wrote a year 1 date into the published json.
    reach = _reach(fit, 1.0) if dated else None
    lo = min(r["x"] for r in plotted)
    hi = max(r["x"] for r in plotted)
    # The timeline runs out until the low edge of the band has reached 1.00 too, so the interval the
    # chart quotes is not just the interval that fits on it. Five years is the stop, since a line
    # this shallow can take forever. The index chart stays inside its data.
    if dated and reach:
        far = np.arange(lo, lo + 5 * 365, 7.0)
        _, _, hi_far = _curve(fit, far)
        crossed = far[hi_far >= 1.0]
        # Past the date the line itself reaches 0.95, or the arrow that names that date is drawn
        # outside the axis and never appears.
        end = max(crossed[0] if len(crossed) else far[-1], reach) + 0.04 * (hi - lo)
        # Room to the left of the oldest model, or its label is placed outside the axis and lands on
        # the y title: it is pinned to the edge and the placer has nowhere else to put it.
        span = (lo - 0.10 * (hi - lo), float(end))
    else:
        span = (lo - 0.04 * (hi - lo), hi + 0.04 * (hi - lo))
    grid = np.linspace(*span, 400)
    line, band_lo, band_hi = _curve(fit, grid)
    # The date chart needs room above the 1.00 line for the crossing arrow, which is drawn 46 pixels
    # above it and otherwise lands in the header.
    top = max(1.14 if dated else 1.06, max(r["score"] for r in plotted) + 0.08)
    # The date chart carries wassname's son, so its floor is 0. The page's own chart does the same,
    # and the two have to agree.
    floor = 0.0 if dated else min(min(r["score"] for r in plotted) - 0.09, float(band_lo.min()))
    # A date axis reads dates, so everything x goes over as a day string. Label placement stays in
    # ordinals, which is the unit the ranges below are in.
    def out(values):
        return [date.fromordinal(int(round(v))).isoformat() for v in values] if dated else list(values)

    fig = go.Figure()
    # The band first, so the points and the line sit on top of it.
    fig.add_scatter(x=out(grid) + out(grid)[::-1], y=list(band_hi) + list(band_lo[::-1]),
                    fill="toself", fillcolor="rgba(107,114,128,0.13)", mode="lines",
                    line={"width": 0}, hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=out(grid), y=list(line), mode="lines", hoverinfo="skip", showlegend=False,
                    line={"color": "#4b5563", "width": 1.5, "dash": "dot"})
    fig.add_scatter(
        x=out([r["x"] for r in plotted]), y=[r["score"] for r in plotted],
        mode="markers", showlegend=False,
        marker={"color": "white", "size": _RING_PX, "line": {"color": _RING_INK, "width": 2}},
        hovertext=[
            f"{r['model']}<br>{r['company']}<br>score {r['score']:+.2f}"
            + (f"<br>AA index {r['x']}, and {AA_EFFORT_GAP[r['model']]} of it is thinking effort "
               f"AA gave it and this bench did not"
               if not dated and r["model"] in AA_EFFORT_GAP else "")
            + (f"<br>released {known[r['model'].split(' (')[0]]}" if dated else "")
            + f"<br>{r['score'] - at:+.3f} from the line"
            # In score units for every point drawn, which on the date chart is more points than the
            # fit used: `fit["resid"]` there covers the fitted months only.
            for r, at in zip(plotted, _curve(fit, [r["x"] for r in plotted])[0])],
        hovertemplate="%{hovertext}<extra></extra>")
    fig.add_hline(y=1.0, line={"dash": "dash", "width": 1, "color": "#9ca3af"},
                  annotation_text=("wassname, flat: 1.00 is his own answer" if dated
                                   else "1.00 = wassname's own answer"),
                  annotation_position="top left",
                  annotation_font={"size": 10, "color": "#6b7280"})
    # A taller header than the pareto chart, because the caveat has to fit on the chart itself.
    size = {"fig_w": 1100, "fig_h": 660,
            # The date chart carries a fourth header line, naming the construction, so its header is
            # taller. A short header pushed that line into the plot.
            "margin": {"l": 80, "r": 150, "t": 190 if dated else 160, "b": 70}}
    # Which construction this is, in the header rather than inside the axes: a date fit is the
    # usual shape for a date chart, so a reader who knows METR's would assume this is one, and the
    # label placer does not know about annotations, so anything in the plot lands on a model name.
    subtitle = axis["subtitle"]
    if dated:
        alt = _timeline_alts(table)
        subtitle += (f"<br>Fitted on the {fit['n']} frontier models, of {alt['n_models']} drawn. "
                     f"The score doubles every {alt['half_front']:.0f} months")
    fig.update_layout(
        title=f'{axis["title"]}<br><sub>{subtitle}<br>{len(load_items())} problems from the '
              'research of <a href="https://wassname.org">wassname</a></sub>',
        xaxis={"title": axis["x_title"], "type": "date" if dated else "linear",
               "range": out(span) if dated else list(span)},
        yaxis={"title": "mean score", "range": [floor, top]},
        template="simple_white", width=size["fig_w"], height=size["fig_h"],
        margin=size["margin"])
    for label in place_labels(
            [{"x": r["x"], "y": r["score"], "text": r["model"], "color": _LABEL_INK}
             for r in sorted(plotted, key=lambda r: -r["score"])],
            x_range=span, y_range=(floor, top),
            obstacles=[(r["x"], r["score"]) for r in plotted], **size):
        fig.add_annotation(**{**label, "x": out([label["x"]])[0]})
    fig.add_annotation(
        text=(f"fit: {fit['r']:+.2f} correlation, "
              + (f"doubling every {_doubling_months(fit):.0f} months" if dated
                 else f"{fit['slope']:+.4f} score a {axis['x_unit']}")
              + f"<br>scatter about the line {fit['sd']:.3f}"
              + (f" in log score, over the {fit['n']} frontier models" if dated
                 else f", over {fit['n']} models")),
        # Bottom right, the one corner both charts leave empty: a weak model with a high public
        # index, or a weak model released last month. Top left is the 1.00 line's own label.
        xref="paper", yref="paper", x=0.99, y=0.08, xanchor="right", yanchor="bottom",
        showarrow=False, align="right", font={"size": 11, "color": _LABEL_INK})
    if dated and reach:
        band = [float(q) for q in grid if _curve(fit, q)[1] <= 1.0 <= _curve(fit, q)[2]]
        fig.add_annotation(
            text=f"the line reaches 1.00 in {date.fromordinal(int(reach)):%b %Y}"
                 + (f",<br>and its own uncertainty covers "
                    f"{date.fromordinal(int(min(band))):%b %Y} to "
                    f"{date.fromordinal(int(max(band))):%b %Y}" if band else ""),
            x=out([reach])[0], y=1.0, ax=-40, ay=-46, axref="pixel", ayref="pixel",
            xanchor="right", align="right", showarrow=True, arrowhead=2, arrowwidth=1,
            arrowcolor="#16a34a", font={"size": 11, "color": "#16a34a"})
    fig.add_annotation(text=RUBRIC_VERSION, xref="paper", yref="paper", x=0.99, y=0.02,
                       xanchor="right", yanchor="bottom", showarrow=False,
                       font={"size": 10, "color": "#9ca3af"})
    fig.add_annotation(text=SITE_URL, xref="paper", yref="paper", x=0.0, y=0.02,
                       xanchor="left", yanchor="bottom", showarrow=False,
                       font={"size": 10, "color": "#9ca3af"})
    png = Path(__file__).parent / "docs" / f"{axis['stem']}{suffix}.png"
    fig.write_image(png, scale=2)
    return png, {**fit, "reach": reach and date.fromordinal(int(reach)).isoformat()}


def _per_item(rows: dict[str, dict[str, float]], settled: set[str]) -> None:
    """The other axis: which questions separate models and which are dead weight."""
    from tabulate import tabulate

    by_item: dict[str, list[float]] = {}
    for scores in rows.values():
        for item, value in scores.items():
            by_item.setdefault(item, []).append(value)
    table = [
        {"item": item, "spread↑": max(values) - min(values), "mean→mid": statistics.mean(values),
         "best↑": max(values), "worst": min(values), "n": len(values),
         "answer": "settled" if item in settled else "*guess*"}
        for item, values in by_item.items()
    ]
    table.sort(key=lambda row: -row["spread↑"])
    print("\n## Per item\n")
    print(tabulate(table, headers="keys", tablefmt="pipe", floatfmt="+.2f"))
    print("\nRanked on `spread↑`, best minus worst. A question near zero spread separates nobody: "
          "either every model gets it or none does. `mean→mid` wants the middle, not either end.")


def _is_judge(billed: str, log) -> bool:
    """Was this model billed for grading rather than answering?

    A panel writes its whole membership into one metadata field, `a+b+c+d+e`, which is equal to no
    single seat's name. Comparing against that string called all five seats candidates, so `$/run`
    read the judge's bill for every model whose own answer replayed from cache, and claude-opus-5
    came out ten times cheaper than glm-5.2.

    Strip the provider prefix on both sides. Panel logs store seats bare (`deepseek/x+google/y`)
    and the older single-judge logs store them prefixed, so comparing raw missed the judge in
    every single-judge log and billed its tokens to the candidate.

    SHOULD: `_usage('logs')['mockllm/gold']` has no `usd` key, since a local mock is never billed.
    ELSE the judge is again counted as the candidate.
    """
    # The metadata lists the whole panel, not the seats that sat. A candidate that is also a seat
    # drops out of its own grading, so its tokens in its own log are always answering. Without
    # this, gemma-4-31b-it, qwen3.7-flash and inkling-small each hid their own cost.
    if billed == log.eval.model:
        return False
    seats = {s.removeprefix("openrouter/") for s in (log.eval.metadata.get("judge") or "").split("+")}
    return billed.removeprefix("openrouter/") in seats


def _usage(log_dir: str) -> dict[str, dict[str, float]]:
    """Per model: what one run of the 12 questions costs in that model's own tokens.

    The judge's bill is not in here. It is roughly the same for every candidate, so folding it in
    compresses the cost axis and lies about the cheap end: gpt-oss-120b answers for $0.005 and read
    $0.72 once the judge was added, which is the wrong answer to "which model is cheap". The judge
    spend is still reported, per run and per role, by `_tokens` below.

    Candidate rows come from every log directory, not just `log_dir`. The inspect cache serves a
    repeat answer free and records no usage, so the only place a candidate's real token count and
    price exists is the log of the run that first paid for it, which is usually another judge's
    directory. Reading `log_dir` alone made every re-judged model read 0 tokens.

    Only logs at the current rubric version count. Reading every version put qwen3.7-flash's
    cheapest-in-the-table bold on a v3 run of a different question set, and inkling's best
    tokens bold on v90. A model whose paid record is older than the current questions reads
    blank, and one cold re-run fills it.

    SHOULD: a model with a full cold run reads `questions == 12`. ELSE it answered the rest from
    cache, and the run is the per-question mean scaled to 12.
    """
    price = _prices()
    root = str(Path(log_dir).parent if Path(log_dir).name != "logs" else log_dir)
    totals: dict[str, dict[str, float]] = {}

    def blank() -> dict[str, float]:
        # None, not 0.0: a run whose answers all came from cache recorded no usage, and a $0.00
        # sorted to the top of the cheap end and got bolded as the best.
        return {"in": None, "out": None, "reasoning": None, "usd": None}

    def cost(billed: str, usage) -> float | None:
        # None, not 0.0, for a model OpenRouter does not list, e.g. a local mockllm variant. A free
        # reading would sort it to the cheap end of the chart and take the bold.
        rate = price.get(billed.split("/", 1)[-1])
        return None if rate is None else usage.input_tokens * rate[0] + usage.output_tokens * rate[1]

    # Candidate tokens and dollars: the newest log at this version that actually recorded them.
    # Any older log is skipped rather than summed, or a model re-run five times reads five runs of
    # tokens. A model whose answers all came from cache has no record anywhere and stays blank.
    # Per question, not per log. A re-run only pays for the questions the cache missed, so a whole
    # log is a whole run only the first time: taking the newest log that recorded anything made a
    # 2-question re-run read as a 12-question run, and gpt-5.6-terra came out 200 times cheaper
    # than gpt-5.6-sol on the same questions. Newest paying log wins per (variant, question), so a
    # model that answered 12 questions across three runs still reads as one run of 12.
    cells: dict[tuple[str, str], dict] = {}
    for log in sorted(_graded_logs(root, any_status=True, with_usage=True),
                      key=lambda log: log.eval.created):
        for sample in log.samples or []:
            # Only an answer that was gradable. A re-run that died partway still records the few
            # tokens it spent, and that would overwrite the real measurement with a fragment.
            scored = sample.scores.get("rubric_judge") if sample.scores else None
            if not scored or not scored.value.get("answered", 1.0):
                continue
            paid = {billed: usage for billed, usage in (sample.model_usage or {}).items()
                    if not _is_judge(billed, log)}
            # Largest record wins, not newest. An answer is written once at full length, and the
            # small records are partial attempts and continuations: minimax-m3 read 20.7 ktok for
            # 12 answers against 200-400 ktok for everyone else, and took the bold for best score
            # per token on it.
            key, size = (_variant(log), sample.id), sum(u.output_tokens for u in paid.values())
            if paid and size > sum(u.output_tokens for u in cells.get(key, {}).values()):
                cells[key] = paid
    by_arm: dict[str, list[dict]] = {}
    for (variant, _), paid in cells.items():
        by_arm.setdefault(variant, []).append(paid)
    # A question answered from cache is in no log, so summing what is there reads low by that
    # question's share. Scale the per-question mean up to the full set instead, and refuse below
    # half: three questions do not price a run, and the questions differ a lot in length.
    n_items = len(load_items())
    for variant, questions in by_arm.items():
        run = totals.setdefault(variant, blank())
        # Kept even when the run is refused, so the report can say which of the two reasons a
        # blank cell has: nothing paid at all, or too little paid to scale from.
        run["questions"] = len(questions)
        if len(questions) * 2 < n_items:
            continue
        scale = n_items / len(questions)
        used = [u for paid in questions for u in paid.values()]
        run["in"] = sum(u.input_tokens for u in used) * scale
        run["out"] = sum(u.output_tokens for u in used) * scale
        run["reasoning"] = sum(u.reasoning_tokens or 0 for u in used) * scale or None
        priced = [cost(b, u) for paid in questions for b, u in paid.items()]
        run["usd"] = None if None in priced else sum(priced) * scale
    return totals


def _tokens(log_dir: str) -> None:
    """Tokens per run, split by who was billed. Multiply by the OpenRouter price per model.

    Only a cold run is priced correctly: anything the inspect cache served cost nothing and is
    not counted here, so a rerun of an unchanged item shows fewer tokens than the first run did.
    """
    from tabulate import tabulate

    price = _prices()
    rows = []
    for log in _graded_logs(log_dir):
        for billed, usage in log.stats.model_usage.items():
            rate = price.get(billed.split("/", 1)[-1], (0.0, 0.0))
            rows.append({
                "run": log.eval.model,
                "billed": billed,
                "role": "judge" if _is_judge(billed, log) else "candidate",
                "input": usage.input_tokens,
                "cached_in": usage.input_tokens_cache_read or 0,
                "output": usage.output_tokens,
                "reasoning": usage.reasoning_tokens or 0,
                "usd": usage.input_tokens * rate[0] + usage.output_tokens * rate[1],
            })
    rows.sort(key=lambda r: (r["run"], r["role"]))
    print("\n### tokens and cost, cold calls only (the cache serves a repeat run free)\n")
    print(tabulate(rows, headers="keys", tablefmt="pipe", intfmt=",", floatfmt=".4f"))
    print(f"\ntotal ${sum(r['usd'] for r in rows):.3f}. A run where the candidate is also the "
          "judge shows one merged row.")


def _estimate(tier: str, log_dir: str) -> None:
    """Projected cost of one cold run per model, from the token profile of a real run.

    Answers "what would it cost to run the frontier list" without running it.
    """
    from tabulate import tabulate

    price = _prices()
    profiles = [
        (log.eval.model, usage)
        for log in _graded_logs(log_dir, any_version=True)
        for billed, usage in log.stats.model_usage.items()
        if billed == log.eval.model
    ]
    assert profiles, f"no run in {log_dir} to take a token profile from"
    measured, usage = profiles[-1]
    # One row of the panel is billed per judge, so the judge cost is summed over the panel and not
    # read off `JUDGE_MODEL`, which is the panel's identity string and not a model anyone bills.
    rates = {m: price[m.removeprefix("openrouter/")] for m in JUDGE_PANEL}
    judge = sum(
        u.input_tokens * rates[billed][0] + u.output_tokens * rates[billed][1]
        for log in _graded_logs(log_dir, any_version=True)
        for billed, u in log.stats.model_usage.items()
        if billed in rates
    ) / max(1, len(profiles))
    rows = []
    for model in MODELS[tier]:
        rate = price[model.split("/", 1)[-1]]
        candidate = usage.input_tokens * rate[0] + usage.output_tokens * rate[1]
        rows.append({"model": model.replace("openrouter/", ""),
                     "run $↑": candidate + judge, "if 2x verbose $": candidate * 2 + judge,
                     "in $/M": rate[0] * 1e6, "out $/M": rate[1] * 1e6})
    rows.sort(key=lambda r: r["run $↑"])
    print(tabulate(rows, headers="keys", tablefmt="pipe", floatfmt=".3f"))
    print(f"\nOne cold pass over {len(load_items())} items, projected from the measured profile of "
          f"{measured} ({usage.input_tokens:,} in, {usage.output_tokens:,} out) plus a fixed "
          f"${judge:.4f} of judge. Total for this tier: ${sum(r['run $↑'] for r in rows):.2f}. "
          "A model that writes more than the profile costs more, hence the 2x column.")


@cache
def _model_records() -> dict[str, dict]:
    """OpenRouter's public model list, keyed by id. No key needed."""
    import json
    import urllib.request

    with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=30) as response:
        return {m["id"]: m for m in json.load(response)["data"]}


def _prices() -> dict[str, tuple[float, float]]:
    """(input, output) USD per token from OpenRouter's public model list. No key needed."""
    return {i: (float(m["pricing"]["prompt"]), float(m["pricing"]["completion"]))
            for i, m in _model_records().items()}


@cache
def _lowest_effort(model: str) -> dict:
    """The quietest thinking rung this model actually lists, as an `extra_body` fragment.

    Sending a rung a model does not have is not an error. The provider silently falls back to its
    own default, so the run measures that default while the report claims the level we asked for.
    That is what happened under a hardcoded `minimal`: 28 of the 32 models in `MODELS["hle"]` do
    not list it, and grok-4.6 (`[xhigh, high, medium, low]`, default `high`) answered a whole table
    at `high` while every chart said `minimal`. Measured 2026-08-22: at a real `low` it writes 1328
    tokens against 5922 under `minimal`.

    A model with no rungs at all keeps an empty dict and runs at its default, because there is
    nothing to ask for. OpenRouter's record is not always right about this, so `EFFORT_OVERRIDE`
    carries the measured rung for the two it gets wrong.

    SHOULD: claude-opus-5 -> low, glm-5.2 -> high (its floor), gpt-4 -> {}. ELSE the roster moved
    or OpenRouter renamed a rung, and the arm is no longer the lowest rung.
    """
    if model.removeprefix("openrouter/") in EFFORT_OVERRIDE:
        return EFFORT_OVERRIDE[model.removeprefix("openrouter/")]
    record = _model_records()[model.removeprefix("openrouter/")]
    listed = (record.get("reasoning") or {}).get("supported_efforts") or []
    ranked = [rung for rung in EFFORT_RUNGS if rung in listed]
    return {"reasoning": {"effort": ranked[0]}} if ranked else {}


@cache
def _quiet_effort(model: str) -> dict:
    """What to send on the continuation turn, which wants prose and not more thinking.

    Three cases, because no single setting works for the whole roster. Most models take
    `{"enabled": false}` and then spend nothing on thinking. A model whose reasoning is mandatory
    400s on it ("Reasoning is mandatory for this endpoint and cannot be disabled") and gets its
    lowest rung instead. `EFFORT_OVERRIDE` holds the two whose record is wrong.
    """
    if model.removeprefix("openrouter/") in EFFORT_OVERRIDE:
        return EFFORT_OVERRIDE[model.removeprefix("openrouter/")]
    record = _model_records()[model.removeprefix("openrouter/")]
    if (record.get("reasoning") or {}).get("mandatory"):
        return _lowest_effort(model)
    return {"reasoning": {"enabled": False}}


@cache
def _provider_prefs(model: str) -> dict:
    """Who may serve this model's answers, decided from its own endpoint list.

    Two facts drive it. Open-weight models are served by dozens of hosts at quantizations from
    bf16 down to fp4, and fp4 is not the model we mean to score. Closed models report `unknown`
    from every endpoint including first-party, so sending the allow-list to them matches nothing
    and OpenRouter answers "No endpoints found for the request with quantization: ...".

    So the constraint is carried by the `order` list rather than by the `quantizations` filter: the
    vendor's own endpoint first, then every endpoint that reports a quantization we accept, and
    nothing else. The filter cannot do this on its own, because it is a hard exclusion and it threw
    out the vendor: deepseek-v4-pro-0813 reports `unknown` from DeepSeek and `fp8` from Novita, so
    the fp8 allow-list removed DeepSeek and Novita served the v96 answers.

    SHOULD: deepseek-v4-pro-0813 comes back with DeepSeek first and no BaseTen, which is its fp4
    endpoint. ELSE the vendor is being filtered out by its own blank field.
    """
    import json
    import urllib.request

    author = model.removeprefix("openrouter/").split("/")[0]
    request = urllib.request.Request(
        f"https://openrouter.ai/api/v1/models/{model.removeprefix('openrouter/')}/endpoints",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
    with urllib.request.urlopen(request, timeout=30) as response:
        endpoints = json.load(response)["data"]["endpoints"]
    # "deepseek" -> "DeepSeek", "z-ai" -> "Z.AI", "moonshotai" -> "Moonshot AI". Match on letters
    # only, which survives every punctuation style OpenRouter uses for a vendor's own name.
    letters = "".join(c for c in author.lower() if c.isalpha())
    first_party = [e["provider_name"] for e in endpoints
                   if "".join(c for c in e["provider_name"].lower() if c.isalpha()).startswith(letters)]
    good = [e["provider_name"] for e in endpoints if e.get("quantization") in GOOD_QUANTS]
    order = list(dict.fromkeys(first_party + [h for h in PREFERRED_HOSTS if h in good] + good))
    # Every endpoint reports `unknown`, which is every closed model. There is nothing to choose
    # between, so the vendor leads and the rest of its own clouds stay reachable.
    if not good:
        # One endpoint means there is nothing to choose between, so the gate only rejects the whole
        # model. o1 is served by OpenAI alone and 404s under it, because OpenRouter does not list
        # max_completion_tokens as an o1 parameter.
        return {"provider": {"order": list(dict.fromkeys(first_party + PREFERRED_HOSTS)),
                             "allow_fallbacks": True,
                             "require_parameters": len(endpoints) > 1}}
    return {"provider": {"order": order, "allow_fallbacks": False, "require_parameters": True}}


def _issues(log_dir: str, judge: str) -> None:
    """Every complaint about an item, from the candidates and from the judge.

    A prompt that several models call ambiguous is a broken item. Fix the prompt, not the score.
    """
    complaints: dict[str, list[str]] = {}
    for log in _latest_logs(log_dir, judge):
        for sample in log.samples or []:
            if not sample.scores:
                continue
            model = log.eval.model.split("/")[-1]
            for line in sample.scores["rubric_judge"].answer.splitlines():
                if (line.strip().startswith("ISSUE:")
                        and not re.match(r"ISSUE:\s*(?:none|no issues?)\b", line.strip(), re.I)):
                    complaints.setdefault(sample.id, []).append(f"[{model} says] {line.strip()}")
            # Every pass, not the first. The passes disagree about what is wrong with an item,
            # and reading one of three threw away most of the judge's side of this report.
            for i, note in enumerate(sample.scores["rubric_judge"].metadata.get("judge_notes", [])):
                complaints.setdefault(sample.id, []).append(f"[judge pass{i} on {model}] {note}")
    for item, lines in sorted(complaints.items()):
        print(f"\n## {item}")
        for line in lines:
            print(f"- {' '.join(line.split())[:400]}")


def _print_grade(grade) -> None:
    """Every rubric point and trap, with the score and quote from each judge pass.

    A mean hides which pass voted why, and the gate lines say which credits were refused.

    `rung` is the rung the pass named for itself before scoring. A rung that says 0.0 next to a
    score of 1.0 is the drift this report exists to show: it was the top finding of round 20's
    audit, on six of thirteen items.
    """
    def show(label: str, point_id: str, rate: float) -> None:
        print(f"  {label}{rate:+.2f} {point_id}")
        for i, (score, lines, quote, rung) in enumerate(grade.metadata["quotes"].get(point_id, [])):
            print(f"       pass{i} {score:.1f} L{lines or '-':<7} {quote[:200]!r}")
            print(f"             rung: {rung[:200]}")
            if score != _rung_score(rung):
                print("             RUNG MISMATCH: this pass contributes 0.0")

    for point_id, rate in grade.metadata["points"].items():
        show("", point_id, rate)
    for point_id, rate in grade.metadata["traps"].items():
        show("TRAP ", point_id, rate)
    if grade.metadata["uncited"]:
        print(f"  refused for no real answer line: {grade.metadata['uncited']}")
    if grade.metadata.get("shared_quotes"):
        print(f"  two points on one span: {grade.metadata['shared_quotes']}")
    print(f"judge_notes: {grade.metadata['judge_notes']}")


def _confusion(log_file: str) -> None:
    """Dump everything needed to sanity check one item: thinking, answer, and how it was graded.

    Read this to tell three failures apart: the model misread the question, the model understood
    it and answered differently, or the judge marked a good answer wrong.
    """
    from inspect_ai.log import read_eval_log

    log = read_eval_log(log_file)
    print(f"# per-item audit: candidate {log.eval.model}, judge {log.eval.metadata['judge']}\n")
    for sample in log.samples or []:
        # Every assistant turn, not just the last: when the out-of-budget retry fires, the
        # thinking is in the first turn and the last turn has none.
        trace = "".join(
            part.reasoning
            for message in sample.messages
            if isinstance(message.content, list)
            for part in message.content
            if getattr(part, "reasoning", None)
        )
        grade = sample.scores["rubric_judge"]
        print(f"\n\n# {sample.id}  score {grade.value['score']:+.2f}\n")
        print(f"## thinking ({len(trace)} chars)\n\n{trace or '(none returned)'}")
        print(f"\n## answer\n\n{grade.answer}")
        print(f"\n## judge said\n\n{grade.explanation}\n")
        _print_grade(grade)


def _validity(log_dir: str, weak: str = "gemma-3-4b-it", judge: str | None = None) -> None:
    """Is an item measuring anything? Report difficulty, weak-vs-rest separation, and spread."""
    from tabulate import tabulate

    by_item: dict[str, dict[str, float]] = {}
    noise: dict[str, list[float]] = {}
    by_point: dict[str, dict[str, list[float]]] = {}
    lengths: list[tuple[int, float]] = []
    for log in _latest_logs(log_dir, judge):
        model = log.eval.model.split("/")[-1]
        for sample in log.samples or []:
            score = sample.scores["rubric_judge"].value if sample.scores else None
            if score and score.get("answered", 1.0):
                by_item.setdefault(sample.id, {})[model] = score["score"]
                lengths.append((len(sample.scores["rubric_judge"].answer), score["score"]))
                meta = sample.scores["rubric_judge"].metadata
                if meta.get("pass_scores"):
                    # Raw, so this is disagreement before the anchor map. The map moves a seat by
                    # under 0.02, so it does not change what counts as noise here.
                    noise.setdefault(sample.id, []).append(statistics.pstdev(meta["pass_scores"]))
                for point_id, rate in meta["points"].items():
                    by_point.setdefault(sample.id, {}).setdefault(point_id, []).append(rate)
    rows = []
    for item, scores in sorted(by_item.items()):
        rest = [v for m, v in scores.items() if m != weak]
        weak_score = scores.get(weak)
        flags = []
        if rest and max(rest) < 0.05:
            flags.append("dead: no model scores")
        if rest and min(rest) > 0.90:
            flags.append("saturated: everyone scores")
        if weak_score is not None and rest and weak_score >= max(rest):
            flags.append(f"weak model ({weak}) ties or wins")
        # Max, not mean: on CW#2 three models had near-zero pass spread and the
        # fourth had 0.07, which a mean averaged down to 0.02 and called a real defect noise.
        judge_sd = max(noise[item]) if noise.get(item) else None
        spread = max(scores.values()) - min(scores.values())
        if judge_sd and spread < 2 * judge_sd:
            flags.append("spread inside judge noise")
        rows.append({
            "item": item, "n": len(scores),
            "mean": statistics.mean(scores.values()),
            "min": min(scores.values()), "max": max(scores.values()),
            "sd": statistics.pstdev(scores.values()),
            "judge_sd": judge_sd,
            "weak": weak_score, "flags": "; ".join(flags),
        })
    rows.sort(key=lambda r: r["mean"])
    print(tabulate(rows, headers="keys", tablefmt="pipe", floatfmt="+.2f"))
    print("\nAn item wants a mid mean, a spread above judge noise, and the weak model last.")
    print("`judge_sd` is the spread over single-pass scores: what one dissenting judge pass is "
          "worth. A defect that moves a score less than this is inside the noise the passes exist "
          "to absorb.")
    # An item can look healthy while a third of its weight is unreachable, which is how
    # KL#3 ended up with two weight-2 points nobody scored.
    dead = [f"{item}.{point}" for item, points in sorted(by_point.items())
            for point, rates in sorted(points.items()) if max(rates) < 0.05]
    if dead:
        print(f"\nrubric points no model in these logs reaches ({len(dead)}): {', '.join(dead)}")
        print("Either the point is unfair, or the prompt never asks for it. Check it is derivable "
              "from the prompt alone before blaming the models. Treat this as an inspection flag.")
    # Every answer is capped at the same word count, so a judge that pays for length rather than
    # content shows up here. Above about +0.5 the bench is measuring verbosity.
    if len(lengths) > 4:
        chars, scores_ = zip(*lengths)
        print(f"\nscore vs answer length: r = {statistics.correlation(chars, scores_):+.2f} "
              f"over {len(lengths)} answers (want |r| < 0.5)")


def _calibrate(
    judge_model: str, passes: int, temperature: float, judge_max_tokens: int, log_dir: str,
    items: str = "*.md",
) -> None:
    """Judge validity: the gold answer must score high, another item's gold answer must score low.

    A low ceiling means the rubric or the judge is broken. A high floor means the judge is rubber
    stamping. Neither is a fact about any model under test.
    """
    import sys

    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import ModelOutput
    from tabulate import tabulate

    item_pattern = items
    items = load_items(item_pattern)
    gold = {item["id"]: item["gold_answer"] for item in items}
    ids = list(gold)
    variants = {
        "gold": gold,
        "offtopic": {i: gold[ids[(n + 1) % len(ids)]] for n, i in enumerate(ids)},
    }
    scores: dict[str, dict[str, float]] = {}
    for variant, texts in variants.items():
        async def mock(input, tools, tool_choice, config, texts=texts) -> ModelOutput:
            prompt = input[0].text
            item = next(i for i in items if i["prompt"][:80] in prompt)
            return ModelOutput.from_content("mockllm", texts[item["id"]])

        log = inspect_eval(
            wassname_ml_bench(items=item_pattern, judge_model=judge_model, judge_passes=passes,
                              judge_temperature=temperature, judge_max_tokens=judge_max_tokens,
                              cache_scope=f"calibration-{variant}-{RUBRIC_VERSION}"),
            model=get_model(f"mockllm/{variant}", custom_outputs=mock),
            log_dir=log_dir,
            display="none",
        )[0]
        scores[variant] = {s.id: s.scores["rubric_judge"].value["score"] for s in log.samples if s.scores}

    rows = [
        {"item": i, "gold": scores["gold"].get(i), "offtopic": scores["offtopic"].get(i),
         "gap": (scores["gold"].get(i) or 0) - (scores["offtopic"].get(i) or 0)}
        for i in ids
    ]
    rows.sort(key=lambda r: r["gap"])
    print(f"# judge calibration: {judge_model}, {passes} passes\n", file=sys.stderr)
    print(tabulate(rows, headers="keys", tablefmt="pipe", floatfmt="+.2f"), file=sys.stderr)
    ceiling = statistics.mean(r["gold"] for r in rows if r["gold"] is not None)
    floor = statistics.mean(r["offtopic"] for r in rows if r["offtopic"] is not None)
    print(f"\nmean ceiling {ceiling:+.2f} (want >= +0.80), mean floor {floor:+.2f} (want <= +0.10)", file=sys.stderr)
    print("An item with a low gold score has a broken rubric or an unreadable gold answer.", file=sys.stderr)


def _smoke() -> None:
    """Offline: the candidate must not see the reference answer, and the score math must hold."""
    import os
    import tempfile

    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ModelOutput

    items = load_items()
    assert items, "no items in items/"
    item = items[0]
    expected_student_inputs = {item["prompt"] + EXIT_INTERVIEW for item in items}
    seen_student_inputs = []
    continuation_seen = []

    judge_calls: dict[str, int] = {}

    # One item is cut off mid-answer, so the continuation turn runs in the smoke and the graded
    # answer must come back whole. Without this the truncation path is never exercised offline.
    # Picked by position, not by name, because item ids may not appear in the published bench.py.
    CUT_ITEM = items[0]["id"]
    SMOKE_QUIET = {"reasoning": {"enabled": False}}
    HALF, REST = ANSWER_UNDER_TEST[:20], ANSWER_UNDER_TEST[20:]

    async def mock_candidate(input, tools, tool_choice, config) -> ModelOutput:
        prompt = input[0].text
        if "length limit mid-answer" in input[-1].text:
            # The continuation asks for prose only, and carries the cut answer for context.
            assert config.max_tokens == FINAL_ANSWER_MAX_TOKENS, config.max_tokens
            assert config.extra_body == SMOKE_QUIET, config.extra_body
            continuation_seen.append(prompt)
            return ModelOutput.from_content("mockllm", REST)
        assert prompt in expected_student_inputs, "candidate received more than the parsed prompt"
        seen_student_inputs.append(prompt)
        assert config.max_tokens == ANSWER_MAX_TOKENS, config.max_tokens
        if next(i["id"] for i in items if i["prompt"][:80] in prompt) == CUT_ITEM:
            return ModelOutput(model="mockllm", choices=[ChatCompletionChoice(
                message=ChatMessageAssistant(content=HALF, model="mockllm"),
                stop_reason="max_tokens")])
        return ModelOutput.from_content("mockllm", ANSWER_UNDER_TEST)

    async def mock_judge(input, tools, tool_choice, config) -> ModelOutput:
        prompt = input[0].text
        graded = next(i for i in items if i["prompt"][:80] in prompt)
        judge_calls[graded["id"]] = judge_calls.get(graded["id"], 0) + 1
        # Which rung of the retry ladder this is, read off the conversation rather than off a
        # call counter. The panel grades its seats and passes concurrently, so two ladders for one
        # item are in flight at once and a counter interleaves them into nonsense.
        if "Continue grading the numbered candidate answer" in input[-1].text:
            assert ANSWER_UNDER_TEST in input[0].text, "judge retry lost the candidate answer"
            return ModelOutput.from_content("mockllm", "{}")
        if "failed validation" not in input[-1].text:
            return ModelOutput(model="mockllm", choices=[ChatCompletionChoice(
                message=ChatMessageAssistant(content="", model="mockllm"), stop_reason="max_tokens")])
        assert ANSWER_UNDER_TEST in input[0].text, "judge correction lost the candidate answer"
        assert graded["gold_answer"][:80] in prompt, "judge must see the reference answer"
        assert ANSWER_UNDER_TEST in prompt, "judge must see the candidate answer"
        # One field per rubric id, the shape grade_model() builds.
        grade = {"evidence": "fixture", "judge_note": ""}
        # Every point but the last cites the answer.
        for point in graded["rubric"][:-1]:
            grade[point["id"]] = {"lines": "1", "rung": "1.0 fixture", "score": 1.0}
        # Last point cites a line that does not exist, so it must score zero.
        grade[graded["rubric"][-1]["id"]] = {
            "lines": "999", "rung": "1.0 fixture", "score": 1.0,
        }
        for trap in graded.get("traps", []):
            grade[trap["id"]] = {"lines": "", "rung": "", "score": 0.0}
        # Half the beyond-reference rung, so the fixture proves an item can score above 1.0.
        grade[BEYOND_ID] = {"lines": "1", "rung": "0.5 fixture", "score": 0.5}
        return ModelOutput.from_content("mockllm", json.dumps(grade))

    with tempfile.TemporaryDirectory() as tmp:
        os.environ["INSPECT_CACHE_DIR"] = f"{tmp}/cache"
        log = inspect_eval(
            wassname_ml_bench(
                judge_model=get_model("mockllm/judge", custom_outputs=mock_judge),
                judge_passes=2,
                quiet=SMOKE_QUIET,
            ),
            model=get_model("mockllm/candidate", custom_outputs=mock_candidate),
            log_dir=f"{tmp}/logs",
            display="none",
        )[0]
        assert log.status == "success", [(sample.id, sample.error) for sample in log.samples]
        assert set(seen_student_inputs) == expected_student_inputs
        by_id = {s.id: s for s in log.samples}
        for graded in items:
            weights = [p["weight"] for p in graded["rubric"]]
            expected = (sum(weights) - weights[-1]) / sum(weights) + BEYOND_WEIGHT * 0.5
            got = by_id[graded["id"]].scores["rubric_judge"].value["score"]
            assert abs(got - expected) < 1e-9, f"{graded['id']}: {got} != {expected}"
            points = by_id[graded["id"]].scores["rubric_judge"].metadata["points"]
            assert points[BEYOND_ID] == BEYOND_WEIGHT * 0.5, points[BEYOND_ID]
        assert all(calls == 6 for calls in judge_calls.values()), judge_calls
        # A cut-off answer is continued and graded whole, not scored 0 for hitting the cap.
        assert len(continuation_seen) == 1, continuation_seen
        cut = by_id[CUT_ITEM].scores["rubric_judge"]
        assert cut.value["answered"] == 1.0, cut.explanation
        assert cut.answer == ANSWER_UNDER_TEST, cut.answer
        # A continuation that reopens a fence inside an open one must not paste the fence back in.
        assert _rejoin("text ```py\ncode", "```py\nmore") == "more"
        assert _rejoin("text ```py\ncode```", "```py\nmore") == "```py\nmore"
        # The panel: one company per seat, so a judge sitting out its own family still leaves four.
        seats = [_company_of(m) for m in JUDGE_PANEL]
        assert len(set(seats)) == len(seats), f"two seats from one company: {seats}"
        # The mock judge above is a single model, which runs the lone-judge path and reaches
        # neither rule below. A candidate whose company holds a seat loses exactly that one.
        sat = [j for j in JUDGE_PANEL if not _sits_out(j, "openrouter/openai/gpt-5.6-sol")]
        assert sat == [j for j in JUDGE_PANEL if "openai/" not in j], sat
        assert [j for j in JUDGE_PANEL if not _sits_out(j, "openrouter/minimax/minimax-m3")] \
            == list(JUDGE_PANEL), "a candidate with no seat of its own keeps all five"
        # Seat leniency: one seat marks every answer 0.1 above the rest, and the fit has to find
        # that and nothing else. Sum-to-zero puts it at +0.08 against -0.02 for the other four.
        synthetic = [(f"answer{a}", seat, 0.5 + a * 0.1 + (0.1 if seat == "lenient" else 0.0))
                     for a in range(6)
                     for seat in ("lenient", "b", "c", "d", "e")]
        effects = _seat_effects(synthetic)
        assert abs(effects["lenient"] - 0.08) < 1e-6, effects
        assert all(abs(effects[s] + 0.02) < 1e-6 for s in "bcde"), effects
        missing = [m for m in JUDGE_PANEL if m.removeprefix("openrouter/") not in JUDGE_ANCHORS]
        assert not missing, f"no measured anchors, run `just calibrate` for: {missing}"
        for name, (off, gold) in JUDGE_ANCHORS.items():
            assert gold - off > 0.5, f"{name} cannot tell gold from off-topic: {off} to {gold}"
            assert abs(_on_anchors(name, gold) - 1.0) < 1e-9, "gold must map to 1.0"
            assert abs(_on_anchors(name, off)) < 1e-9, "off-topic must map to 0.0"
        try:
            PointCheck(lines="1", rung="1.0 fixture", score=2.0)
        except ValidationError:
            pass
        else:
            raise AssertionError("judge score outside 0.0, 0.5, 1.0 was accepted")
        PointCheck(lines="1", rung="1.0: full credit", score=1.0)
        assert _rung_score("1.0, full credit") == 1.0
        assert _rung_score("0.5 partial credit") == 0.5
        assert _rung_score("") == 0.0
        # One or two seats of five may drop out of a cell; three cannot, and a lone judge cannot.
        assert [_panel_survives(ok, 5) for ok in (5, 4, 3, 2)] == [True, True, True, False]
        assert not _panel_survives(0, 1) and _panel_survives(1, 1)
        # A fenced reply is the same grade with ```json around it.
        assert _validate(
            PointCheck, '```json\n{"lines": "1", "rung": "1.0 x", "score": 1.0}\n```').score == 1.0
        # A refused cell borrows from the same company's best complete model, never from a rival
        # and never from a model that is itself short.
        rows = {"openrouter/anthropic/claude-fable-5": {"a": 0.1},
                "openrouter/anthropic/claude-opus-5": {"a": 0.2, "b": 0.3},
                "openrouter/anthropic/claude-sonnet-5": {"a": 0.9, "b": 0.1},
                "openrouter/openai/gpt-5.6-luna": {"a": 1.0, "b": 1.0},
                "openrouter/deepseek/deepseek-v4-flash": {"a": 0.5}}
        got = _borrow(rows, {("claude-fable-5", "b"), ("deepseek-v4-flash", "b")}, n_items=2)
        assert got == {"openrouter/anthropic/claude-fable-5":
                       ("openrouter/anthropic/claude-sonnet-5", {"b"})}, got
        assert rows["openrouter/anthropic/claude-fable-5"] == {"a": 0.1, "b": 0.1}, rows
        # A run at a non-default effort is its own variant, or its answers merge into the bare row.
        # Both parts show, so a skill run at a raised effort is not filed as the skill variant.
        high = {"reasoning": {"effort": "high"}}
        assert wassname_ml_bench().metadata["cache_scope"] is None
        assert wassname_ml_bench(reasoning=high, effort_label="high"
                                 ).metadata["cache_scope"] == "effort:high"
        # The default arm is each model's lowest rung, so it is a bare row even though the config
        # carries a real effort. Without this a resolved rung would split every model into a variant.
        assert wassname_ml_bench(reasoning={"reasoning": {"effort": "low"}}
                                 ).metadata["cache_scope"] is None
        # A budget other than the default is a variant too, or it replays the default's answers.
        assert wassname_ml_bench(max_tokens=40_000).metadata["cache_scope"] == "budget:40000"
        assert wassname_ml_bench(max_tokens=ANSWER_MAX_TOKENS).metadata["cache_scope"] is None
        fake_skill = Path(tempfile.mkdtemp()) / "ml-debug" / "SKILL.md"
        fake_skill.parent.mkdir()
        fake_skill.write_text("fixture skill")
        assert wassname_ml_bench(skill=str(fake_skill), reasoning=high, effort_label="high"
                                 ).metadata["cache_scope"] == "skill:ml-debug effort:high"
        assert _cited("fixture", PointCheck(lines="1,1", rung="1.0", score=1.0),
                      ANSWER_UNDER_TEST, [])
        assert not _cited("fixture", PointCheck(lines="1,1,1", rung="1.0", score=1.0),
                          ANSWER_UNDER_TEST, [])
        binary_grade = grade_model([{
            "id": "binary_fixture", "weight": 1,
            "point": "Fixture. This point is 1.0 or 0.0.",
        }], [])
        assert binary_grade.model_fields["binary_fixture"].annotation is BinaryPointCheck
        try:
            BinaryPointCheck(lines="1", rung="0.5 fixture", score=0.5)
        except ValidationError:
            pass
        else:
            raise AssertionError("binary rubric point accepted partial credit")
        mismatch = PointCheck(lines="1", rung="0.0 miss", score=0.5)
        assert not _score_matches_rung(mismatch)
        assert _consistent_score(mismatch) == 0.0
        assert re.match(r"ISSUE:\s*(?:none|no issues?)\b", "ISSUE: no issue.", re.I)
        print(f"smoke PASS: {len(items)} items, {log.results.total_samples} samples, "
              f"student input is exactly one parsed prompt plus exit interview, "
              f"candidate and judge retries retain their subject, "
              f"invalid point scores reject, rung mismatches contribute zero, "
              f"a hit citing a line that does not exist scores zero, "
              f"weights add up\n"
              f"  items: {', '.join(i['id'] for i in items)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="candidate model, e.g. openrouter/openai/gpt-5.1")
    parser.add_argument("--items", default="*.md", help="glob over items/")
    # Comma separated. One name calibrates or reads one judge's logs; the default is the panel.
    parser.add_argument("--judge-model", default=",".join(JUDGE_PANEL),
                        type=lambda s: tuple(s.split(",")))
    parser.add_argument("--judge-passes", type=int, default=JUDGE_PASSES)
    parser.add_argument("--judge-temperature", type=float, default=JUDGE_TEMPERATURE)
    parser.add_argument("--judge-max-tokens", type=int, default=JUDGE_MAX_TOKENS)
    parser.add_argument("--reasoning", default="off",
                        help="off, none (send no effort at all), or an OpenRouter effort level")
    parser.add_argument("--max-tokens", type=int, default=ANSWER_MAX_TOKENS,
                        help="answer budget. Only to sit under a model's own ceiling, like gpt-4's "
                             "4096: lowering it for a model that would use more changes its score")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--dry-run", action="store_true", help="print the call count, spend nothing")
    parser.add_argument("--smoke", action="store_true", help="offline self-test, no API key")
    parser.add_argument("--trace", metavar="LOG.eval", help="print one graded item in full")
    parser.add_argument("--sample", default=None, help="item id for --trace")
    parser.add_argument("--results", action="store_true", help="table over every log in --log-dir")
    parser.add_argument("--calibrate", action="store_true", help="judge validity: gold vs off-topic")
    parser.add_argument("--validity", action="store_true", help="item difficulty and separation")
    parser.add_argument("--issues", action="store_true", help="candidate and judge complaints per item")
    parser.add_argument("--confusion", metavar="LOG.eval", help="per item: thinking, answer, grade")
    parser.add_argument("--estimate", metavar="TIER", help=f"projected cost: {'|'.join(MODELS)}")
    parser.add_argument("--models", metavar="TIER", help="print the model ids in a tier, one a line")
    parser.add_argument("--skill", default=None, metavar="SKILL.md",
                        help="prepend this document to every question, as an uplift variant with its "
                             "own row, e.g. ~/.agents/skills/ml-debug/SKILL.md")
    parser.add_argument("--incomplete", action="store_true",
                        help="print the models that answered fewer than every question, one a line")
    parser.add_argument("--draws", type=int, default=1, metavar="N",
                        help="answer every question N times and average, instead of once. The "
                             "draw index is the seed and is in the cache key, so draws already "
                             "paid for replay free and only the new ones cost. Re-answering is "
                             "the second largest term in a row's error bar, and the only one "
                             "this can shrink")
    args = parser.parse_args()

    if args.smoke:
        _smoke()
    elif args.trace:
        _trace(args.trace, args.sample)
    elif args.results:
        # One writer for every artifact. `_pareto` writes docs/pareto.png as it goes, so leaving
        # docs/results.md to a shell tee let the markdown sit a rubric version behind the chart
        # beside it, which read as models disappearing.
        report = io.StringIO()
        with contextlib.redirect_stdout(report):
            _results(args.log_dir, _judge_id(args.judge_model))
        docs = Path(__file__).parent / "docs"
        # Named after the judge, like the json and the pngs. Writing the markdown unsuffixed let a
        # non-default judge overwrite docs/results.md while docs/results.json kept the default
        # judge's numbers, and publish.sh ships that pair.
        suffix = _report_suffix(_judge_id(args.judge_model))
        (docs / f"results{suffix}.md").write_text(report.getvalue())
        # Dated and versioned, so a month-to-month diff never compares two rubrics by accident.
        (docs / "history").mkdir(exist_ok=True)
        stamp = f"{date.today()}_{RUBRIC_VERSION}{suffix}"
        (docs / "history" / f"{stamp}_results.md").write_text(report.getvalue())
        # The json too, or a month-to-month delta has to be parsed back out of a markdown table.
        (docs / "history" / f"{stamp}_results.json").write_text(
            (docs / f"results{suffix}.json").read_text())
        print(report.getvalue(), end="")
    elif args.validity:
        _validity(args.log_dir, judge=_judge_id(args.judge_model))
    elif args.issues:
        _issues(args.log_dir, _judge_id(args.judge_model))
    elif args.confusion:
        _confusion(args.confusion)
    elif args.models:
        print("\n".join(MODELS[args.models]))
    elif args.incomplete:
        print("\n".join(_incomplete_models(args.log_dir, _judge_id(args.judge_model))))
    elif args.estimate:
        _estimate(args.estimate, args.log_dir)
    elif args.calibrate:
        # One judge at a time: this is what measures a judge's own two anchors.
        for judge in args.judge_model:
            _calibrate(judge, args.judge_passes, args.judge_temperature, args.judge_max_tokens,
                       args.log_dir + "/calibration", args.items)
    else:
        from inspect_ai import eval as inspect_eval

        n = len(load_items(args.items))
        print(f"{n} items x {args.draws} draws x (1 answer + {args.judge_passes} judge calls) = "
              f"{n * args.draws * (1 + args.judge_passes)} calls, minus whatever the cache holds")
        if not args.dry_run:
            # "none" sends no reasoning key at all, for a model that has no such setting: with
            # require_parameters on, asking gpt-4 for an effort matches no endpoint and 404s.
            reasoning = ({} if args.reasoning == "none" else REASONING if args.reasoning == "off"
                         else {"reasoning": {"effort": args.reasoning}})
            # Every candidate in one process, all in flight together. inspect keys its connection
            # semaphore on (provider, key, model), so each candidate provider sees its own 8 and
            # the five judges see one pool each however many candidates are running. One model at
            # a time left every provider but one idle and made a sweep take hours.
            # Models that share a serving policy share an eval call, so the run keeps its
            # parallelism. In practice that is two groups: those with a quantization to filter on
            # and those without. Grouped on the name, before it becomes a Model.
            # Keyed on the serving policy and on the resolved thinking rung, because one eval call
            # shares one GenerateConfig and two models rarely have the same lowest rung.
            groups: dict[tuple[str, str], list] = {}
            for name in args.model.split(","):
                resolved = _lowest_effort(name) if reasoning == REASONING else reasoning
                key = (json.dumps(_provider_prefs(name), sort_keys=True),
                       json.dumps(resolved, sort_keys=True),
                       json.dumps(_quiet_effort(name), sort_keys=True))
                groups.setdefault(key, []).append(_openrouter_model(name))
            # Draws outermost, so every model gets its second answer before any gets its third.
            # A provider having a bad hour then hits one draw of everything rather than every draw
            # of one model, where it would read as that model being worse.
            for draw in range(1, args.draws + 1):
                for (prefs, resolved_json, quiet_json), batch in groups.items():
                    logs = inspect_eval(
                        wassname_ml_bench(args.items, args.judge_model, args.judge_passes,
                                          args.judge_temperature, json.loads(resolved_json),
                                          json.loads(quiet_json), args.judge_max_tokens,
                                          skill=args.skill, draw=draw, provider=json.loads(prefs),
                                          max_tokens=args.max_tokens,
                                          effort_label=None if reasoning == REASONING
                                          else reasoning["reasoning"]["effort"]),
                        model=batch,
                        log_dir=args.log_dir,
                        log_model_api=True,
                        max_tasks=len(batch),
                        display="plain",
                    )
                    for log in logs:
                        print(f"draw {draw} status={log.status} {log.eval.model} log: {log.location}")
