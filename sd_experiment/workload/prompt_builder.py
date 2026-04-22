"""Agent-style prompt construction for SD trade-off experiments.

The goal is to produce prompts that resemble a coding agent mid-task, not
a short instruction-following prompt.  Specifically:

  - System message: agent persona + a realistic-looking tool list.
  - User message: issue description + prior exploration context
    (problem_statement, hints_text, a synthesized partial file listing,
    and — optionally — filler that pads to a target context length).
  - Assistant starts inside `<think>` so the first decoded token is
    reasoning rather than a tool call.  This keeps decoding workloads
    comparable across samples.

Only Llama-3.1 chat template tokens are used.  We deliberately **do not**
rely on `tokenizer.apply_chat_template` — different tokenizer versions
surface different defaults (e.g. `date_string`) and a raw string
assembles more reproducibly.
"""

from __future__ import annotations

import textwrap
from typing import Any

# Llama-3.1 chat template literals.  These match
# meta-llama/Llama-3.1-8B-Instruct/tokenizer_config.json exactly.
BOS = "<|begin_of_text|>"
EOT = "<|eot_id|>"
SH = "<|start_header_id|>"
EH = "<|end_header_id|>"


AGENT_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are an autonomous coding agent working on an open-source Python
    repository.  A human has filed a bug/feature request.  Your job is to
    (1) think step by step about root cause, (2) inspect the repository
    via the provided tools, (3) propose a minimal patch, and
    (4) verify the patch against the failing tests.

    Always reason inside <think>...</think> before emitting any tool call
    or patch.  Keep reasoning concise but explicit: state hypotheses,
    evidence needed, and the next action.  When you emit a tool call,
    use exactly one JSON object per tool call, wrapped in
    <tool_call>...</tool_call>.  Do not invent files that have not
    appeared in tool output.
    """
).strip()


TOOL_DEFINITIONS = textwrap.dedent(
    """\
    Available tools (JSON schemas, one per line):

    {"name":"read_file","params":{"path":"string","start_line":"int|null","end_line":"int|null"}}
    {"name":"grep","params":{"pattern":"string","path":"string","ignore_case":"bool"}}
    {"name":"list_dir","params":{"path":"string","recursive":"bool"}}
    {"name":"apply_patch","params":{"unified_diff":"string"}}
    {"name":"run_tests","params":{"test_selectors":"list[string]"}}
    {"name":"submit","params":{"summary":"string"}}
    """
).strip()


# ---------------------------------------------------------------------------
# Synthesized "prior exploration" blocks.
#
# We want to simulate an agent that has already read a few files and
# accumulated context.  Real SWE-Bench samples do not carry file contents
# in the HF dataset, so we fabricate a plausible-looking exploration log
# from the sample's repo + hints.  This keeps tokens count-realistic
# without requiring git clones.
# ---------------------------------------------------------------------------
_PRIOR_EXPLORATION_HEADER = textwrap.dedent(
    """\
    [prior exploration — turns 1..N elided, summary only]

    Tool: list_dir(path=".", recursive=false)
    Result:
      CHANGELOG.rst  CONTRIBUTING.md  LICENSE  README.rst  docs/
      setup.cfg  setup.py  src/  tests/  tox.ini

    Tool: list_dir(path="src", recursive=true)
    Result (truncated):
    """
).strip()


def _fake_file_listing(repo: str, n_files: int = 40) -> str:
    """Produce a synthetic but stable listing proportional to n_files."""
    # Stable enough across runs with the same (repo, n_files).
    pkg = repo.split("/")[-1].replace("-", "_").lower()
    lines = []
    for i in range(n_files):
        if i == 0:
            lines.append(f"  src/{pkg}/__init__.py")
        elif i < 10:
            lines.append(f"  src/{pkg}/core_{i:02d}.py")
        elif i < 25:
            lines.append(f"  src/{pkg}/handlers/handler_{i:02d}.py")
        else:
            lines.append(f"  tests/test_{pkg}_{i:02d}.py")
    return "\n".join(lines)


def _filler_block(n_blocks: int = 1) -> str:
    """Low-entropy filler that simulates a long exploration transcript.

    Used to reach a target context length when padding is requested.
    Content is intentionally boring so it does not bias acceptance rates
    (tokenizer should coalesce these into common tokens).
    """
    block = textwrap.dedent(
        """\
        Tool: grep(pattern="def __init__", path="src", ignore_case=false)
        Result: 42 matches (truncated)
          src/pkg/core_01.py:12:   def __init__(self, config):
          src/pkg/core_02.py:34:   def __init__(self, *args, **kwargs):
          src/pkg/handlers/handler_10.py:18: def __init__(self, registry):
          ... (many similar) ...

        Tool: read_file(path="src/pkg/core_01.py", start_line=1, end_line=80)
        Result:
          '''Core module.'''
          from __future__ import annotations

          class CoreObject:
              def __init__(self, config):
                  self._config = config
                  self._cache = {}

              def process(self, item):
                  key = item.key
                  if key in self._cache:
                      return self._cache[key]
                  value = self._compute(item)
                  self._cache[key] = value
                  return value
        """
    ).strip()
    return "\n\n".join([block] * n_blocks)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_chat_prompt(
    system: str,
    user: str,
    assistant_prefix: str = "<think>\n",
) -> str:
    """Assemble a Llama-3.1 chat prompt without calling the tokenizer.

    The assistant turn is left open (no EOT) so the model continues it.
    """
    parts = [
        BOS,
        f"{SH}system{EH}\n\n{system}{EOT}",
        f"{SH}user{EH}\n\n{user}{EOT}",
        f"{SH}assistant{EH}\n\n{assistant_prefix}",
    ]
    return "".join(parts)


def build_agent_prompt(
    sample: dict[str, Any],
    *,
    n_fake_files: int = 40,
    n_filler_blocks: int = 0,
) -> str:
    """Build an agent-style Llama-3.1 chat prompt from a SWE-Bench sample.

    Parameters
    ----------
    sample:
        A dict with at least `instance_id`, `repo`, `problem_statement`.
        `hints_text` is optional.
    n_fake_files:
        Number of fabricated paths in the synthetic listing.  Larger values
        lengthen the prompt.
    n_filler_blocks:
        Number of filler "exploration" blocks to append.  Used by the
        bucket-fitting logic to reach a target context length.

    Returns
    -------
    A string ready to hand to `engine.add_request(..., prompt=...)`.
    """
    repo = sample.get("repo", "owner/repo")
    instance_id = sample.get("instance_id", "unknown-instance")
    problem = sample.get("problem_statement", "").strip()
    hints = (sample.get("hints_text") or "").strip()

    listing = _fake_file_listing(repo, n_files=n_fake_files)
    exploration = f"{_PRIOR_EXPLORATION_HEADER}\n{listing}"
    if n_filler_blocks > 0:
        exploration = exploration + "\n\n" + _filler_block(n_filler_blocks)

    user_msg_parts = [
        f"# Task: fix issue `{instance_id}` in repo `{repo}`",
        "",
        "## Issue description",
        problem if problem else "(no problem_statement in sample)",
    ]
    if hints:
        user_msg_parts += ["", "## Maintainer hints", hints]
    user_msg_parts += [
        "",
        "## Prior exploration (this session)",
        exploration,
        "",
        "## Your next turn",
        "Think step by step, decide the next tool call, then produce it.",
    ]
    user_msg = "\n".join(user_msg_parts)

    system_msg = f"{AGENT_SYSTEM_PROMPT}\n\n{TOOL_DEFINITIONS}"
    return build_chat_prompt(system=system_msg, user=user_msg)


def estimate_prompt_length(
    sample: dict[str, Any],
    tokenizer,
    *,
    n_fake_files: int = 40,
    n_filler_blocks: int = 0,
) -> int:
    """Tokenize the assembled prompt and return its length in tokens."""
    text = build_agent_prompt(
        sample,
        n_fake_files=n_fake_files,
        n_filler_blocks=n_filler_blocks,
    )
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return len(ids)


def pad_to_bucket(
    sample: dict[str, Any],
    tokenizer,
    target_tokens: int,
    *,
    max_filler_blocks: int = 256,
) -> tuple[str, int]:
    """Return (prompt_text, prompt_token_len) padded up to ~target_tokens.

    The padding is monotone in `n_filler_blocks`, so a simple linear scan is
    sufficient and avoids a binary search whose midpoints would still need
    one tokenize each.  We stop at the first block count whose length is
    >= target_tokens, capped at `max_filler_blocks`.
    """
    n = 0
    text = build_agent_prompt(sample, n_filler_blocks=n)
    length = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    while length < target_tokens and n < max_filler_blocks:
        n += 1
        text = build_agent_prompt(sample, n_filler_blocks=n)
        length = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    return text, length
