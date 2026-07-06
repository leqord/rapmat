"""Progress callback contract.

A progress callback is called as ``callback(current, total, message,
is_log=False)``. Emitters may omit ``is_log``. Receivers must accept it as a
keyword with a default. ``is_log`` marks messages that should also be appended
to a persistent log, not just shown as transient progress.
"""

from typing import Callable

ProgressCallback = Callable[..., None]
