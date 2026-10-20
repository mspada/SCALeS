import re


def natural_sort_key(s) -> list:
    """Sort key ordering embedded numbers numerically, e.g. A1, A2, A10 instead of A1, A10, A2."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]
