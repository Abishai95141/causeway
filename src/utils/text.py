"""
Safe Text Truncation & Variable Canonicalization Utilities

Provides:
- Sentence-boundary-aware truncation so evidence text is never cut
  mid-word or mid-sentence.
- A single ``canonicalize_var_id()`` function used everywhere a
  variable name is converted to a snake_case graph ID.

All components that need to trim evidence content should use these
helpers instead of raw ``[:N]`` slicing.
"""

from __future__ import annotations
import re
from typing import Optional

# Pre-compiled regex for sentence endings (period, !, ?) followed by
# whitespace or end-of-string.  Handles common abbreviations poorly,
# but for evidence snippets this is good enough.
_SENTENCE_END_RE = re.compile(r'[.!?](?:\s|$)')


def truncate_at_sentence_boundary(
    text: str,
    max_chars: int,
    *,
    suffix: str = " [...]",
    min_chars: int = 80,
) -> str:
    """Truncate *text* at a sentence boundary without destroying meaning.

    Parameters
    ----------
    text:
        The text to (possibly) truncate.
    max_chars:
        Maximum character length for the returned string, **including**
        the suffix.  Must be > 0.
    suffix:
        Appended when truncation occurs.  Set to ``""`` to disable.
    min_chars:
        If no sentence boundary is found within this minimum window,
        fall back to a word boundary.  Prevents pathologically short
        results when the first sentence is very long.

    Returns
    -------
    str
        The original text (if it fits) or a truncated version ending at
        a clean boundary with *suffix* appended.

    Examples
    --------
    >>> truncate_at_sentence_boundary("Hello world. Goodbye.", 18)
    'Hello world. [...]'
    >>> truncate_at_sentence_boundary("Short", 100)
    'Short'
    """
    if not text:
        return text

    if len(text) <= max_chars:
        return text

    # Budget for the actual content (excluding suffix)
    budget = max_chars - len(suffix)
    if budget <= 0:
        budget = max_chars  # degenerate case — just hard-cut

    # Search for the last sentence-ending punctuation within budget
    candidate = text[:budget]
    best_end: Optional[int] = None

    for match in _SENTENCE_END_RE.finditer(candidate):
        # match.end() is right after the whitespace following the punctuation
        pos = match.start() + 1  # include the punctuation itself
        if pos >= min_chars:
            best_end = pos

    if best_end is not None:
        return text[:best_end].rstrip() + suffix

    # No sentence boundary found — fall back to last word boundary
    space_idx = candidate.rfind(' ')
    if space_idx > min_chars:
        return text[:space_idx].rstrip() + suffix

    # Last resort: hard-cut at budget (better than nothing)
    return candidate.rstrip() + suffix


def truncate_evidence(
    content: str,
    max_chars: int = 800,
    *,
    suffix: str = " [...]",
) -> str:
    """Convenience wrapper for truncating evidence bundle content.

    Same as :func:`truncate_at_sentence_boundary` but with defaults
    tuned for typical evidence chunks (~800 chars) seen in the Causeway
    retrieval pipeline.
    """
    return truncate_at_sentence_boundary(
        content, max_chars, suffix=suffix,
    )


def truncate_for_context_tracking(
    content: str,
    max_chars: int = 200,
) -> str:
    """Truncate evidence for the ContextManager's evidence summary tracking.

    Shorter than full evidence — used for the ``_evidence_context`` map
    that keeps a lightweight index of what evidence has been seen.
    """
    return truncate_at_sentence_boundary(
        content, max_chars, suffix="...",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Variable ID canonicalization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def canonicalize_var_id(name: str) -> str:
    """Convert a human-readable variable name to a canonical snake_case ID.

    This is the **single source of truth** for variable-ID construction.
    Every place that needs to go from ``"Daily Customer Traffic"`` to
    ``"daily_customer_traffic"`` MUST call this function rather than
    inlining its own regex.

    Rules:
        1. Lowercase the input.
        2. Replace every run of non-alphanumeric characters with ``_``.
        3. Strip leading / trailing underscores.

    Examples
    --------
    >>> canonicalize_var_id("Daily Customer Traffic")
    'daily_customer_traffic'
    >>> canonicalize_var_id("  Revenue ($) ")
    'revenue'
    """
    return _NON_ALNUM_RE.sub("_", name.lower()).strip("_")
