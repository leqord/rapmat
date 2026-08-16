from dataclasses import dataclass
from typing import Callable

# (key, label, priority)
FooterHint = tuple[str, str, int]  

PRIORITY_PINNED = 0


@dataclass(frozen=True)
class KeyBinding:
    keys: tuple[str, ...]
    label: str | Callable[[], str]
    action: Callable[[], None]
    help: str = ""
    enabled: Callable[[], bool] | None = None
    priority: int = 50
    case_sensitive: bool = False

    def matches(self, key: str) -> bool:
        for k in self.keys:
            if len(k) == 1 and k.isalpha() and not self.case_sensitive:
                if key.lower() == k.lower():
                    return True
            elif key == k:
                return True
        return False

    def is_enabled(self) -> bool:
        return self.enabled is None or self.enabled()

    def label_text(self) -> str:
        return self.label() if callable(self.label) else self.label

    def help_text(self) -> str:
        return self.help or self.label_text()

    def key_text(self) -> str:
        return self.keys[0]

    def key_display(self) -> str:
        return _KEY_DISPLAY.get(self.keys[0], self.keys[0])


_KEY_DISPLAY = {
    "delete": "Del",
    "f5": "F5",
    "esc": "Esc",
    "enter": "Enter",
    "tab": "Tab",
    "left": "Left",
    "right": "Right",
    "up": "Up",
    "down": "Down",
    " ": "Space",
}


def dispatch(bindings: list[KeyBinding], key) -> bool:
    if not isinstance(key, str):
        return False
    for binding in bindings:
        if binding.matches(key) and binding.is_enabled():
            binding.action()
            return True
    return False


def footer_hints(bindings: list[KeyBinding]) -> list[FooterHint]:
    return [
        (b.key_display(), b.label_text(), b.priority)
        for b in sorted(bindings, key=lambda b: b.priority)
        if b.is_enabled()
    ]
