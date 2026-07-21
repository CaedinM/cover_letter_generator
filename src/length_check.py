"""Deterministic character-count check for the final letter.

LLMs can't reliably count characters, so the 1600-1900 length target is enforced
here in plain Python rather than by an LLM. Kept in sync with the length the
writer and editor task prompts request.
"""

LENGTH_MIN = 1600
LENGTH_MAX = 1900


def check_length(text: str) -> tuple[bool, str]:
    """Return ``(passed, feedback)`` for ``text``.

    ``feedback`` describes how to fix an out-of-range letter (used to steer the
    editor re-run) or confirms the length when it passes.
    """
    n = len(text)
    if n < LENGTH_MIN:
        return False, (
            f"The letter is {n} characters, which is too SHORT. It must be between "
            f"{LENGTH_MIN} and {LENGTH_MAX} characters. Add roughly {LENGTH_MIN - n} "
            f"characters of substantive, specific content (more detail on an experience "
            f"or the company), without padding or fluff."
        )
    if n > LENGTH_MAX:
        return False, (
            f"The letter is {n} characters, which is too LONG. It must be between "
            f"{LENGTH_MIN} and {LENGTH_MAX} characters. Tighten the prose to cut roughly "
            f"{n - LENGTH_MAX} characters, trimming redundancy while keeping every "
            f"concrete detail and the overall structure."
        )
    return True, f"The letter is {n} characters, within the {LENGTH_MIN}-{LENGTH_MAX} target."
