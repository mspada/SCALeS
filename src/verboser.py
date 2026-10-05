import time
from rich.live import Live
from rich.spinner import Spinner
from contextlib import contextmanager

"""
This class allows to nicely handle verbose statements. How it functions exactly is not very important.
"""


class Verboser:

    def __init__(self, verbose, text):
        self._verbose, self._text, self._spinner, self._live = verbose, text, None, None

    def __enter__(self):
        if self._verbose:
            self._spinner = Spinner('dots', style='green', text=self._text)
            self._live = Live(self._spinner, refresh_per_second=10)
            self._live.__enter__()
        return self

    def update(self, text):
        self._text = text
        if self._verbose:
                self._spinner.text = text

    def terminate(self, text):
         self._text = text
         if self._verbose:
              self._live.update(text)
    
    def __exit__(self, exc_type, exc_value, traceback):
         if self._verbose:
              self._live.__exit__(exc_type, exc_value, traceback)

    
