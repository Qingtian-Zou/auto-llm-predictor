# Copyright 2024-2026 Qingtian Zou
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Verbose LLM wrapper for debugging API prompts and responses.

When verbose mode is enabled (via the ``-v`` CLI flag), wraps
``ChatOpenAI`` instances so that every ``invoke()`` call prints the
full prompt messages and the LLM response to stderr with clear
delimiters for easy identification.
"""

from __future__ import annotations

import sys
from typing import Any


_SEPARATOR = "═" * 60


class VerboseLLM:
    """Transparent wrapper that logs prompts and responses to stderr.

    Delegates attribute *reads* to the underlying LLM so that it
    behaves identically for callers — the only addition is console
    output around ``invoke()`` calls.  Attribute *writes* are kept on
    this wrapper to avoid corrupting the inner pydantic model.
    """

    def __init__(self, llm: Any, label: str = "LLM") -> None:
        self._llm = llm
        self._label = label

    # -- Delegation --------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)

    # -- Core intercept ----------------------------------------------------

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        """Invoke the underlying LLM, printing prompt and response."""
        llm = self._llm
        label = self._label

        # ── Print prompt ──────────────────────────────────────────
        print(f"\n{_SEPARATOR}", file=sys.stderr)
        print(f"  {_SEPARATOR}", file=sys.stderr)
        print(f"  {label} REQUEST", file=sys.stderr)
        print(f"  {_SEPARATOR}", file=sys.stderr)

        if isinstance(messages, (list, tuple)):
            for msg in messages:
                role = getattr(msg, "type", "unknown")
                content = getattr(msg, "content", str(msg))
                print(f"\n  [{role.upper()}]", file=sys.stderr)
                print(f"  {content}", file=sys.stderr)
        else:
            print(f"  {messages}", file=sys.stderr)

        print(f"{_SEPARATOR}\n", file=sys.stderr)
        sys.stderr.flush()

        # ── Invoke the real LLM ───────────────────────────────────
        response = llm.invoke(messages, **kwargs)

        # ── Print response ────────────────────────────────────────
        content = getattr(response, "content", str(response))
        print(f"\n{_SEPARATOR}", file=sys.stderr)
        print(f"  {_SEPARATOR}", file=sys.stderr)
        print(f"  {label} RESPONSE", file=sys.stderr)
        print(f"  {_SEPARATOR}", file=sys.stderr)
        print(f"\n  {content}", file=sys.stderr)
        print(f"\n{_SEPARATOR}\n", file=sys.stderr)
        sys.stderr.flush()

        return response

    # Allow bind_tools to return a new VerboseLLM wrapping the result
    def bind_tools(self, *args: Any, **kwargs: Any) -> VerboseLLM:
        """Bind tools on the underlying LLM and re-wrap the result."""
        bound = self._llm.bind_tools(*args, **kwargs)
        return VerboseLLM(bound, label=self._label)
