"""Shared urwid palette and terminal color depth negotiation.
"""

import os
import sys
from collections.abc import Mapping

# ------------------------------------------------------------------ #
#  Global palette
# ------------------------------------------------------------------ #

PALETTE = [
    ("header", "white", "dark blue", "bold"),
    ("footer", "black", "light gray"),
    ("body", "light gray", "default"),
    ("focus", "white", "dark green", "standout"),
    ("error", "light red", "default"),
    ("success", "light green", "default"),
    ("unconv", "yellow", "default"),
    ("details", "white", "default"),
    ("details_title", "light cyan", "default", "bold"),
    ("section", "yellow", "default", "bold"),
    ("col_header", "white", "dark gray", "bold"),
    ("menu_item", "light gray", "default"),
    ("menu_focus", "white", "dark green", "standout"),
    ("dialog", "white", "dark gray"),
    ("btn_focus", "white", "dark green", "bold"),
    ("dropdown_hl", "white", "dark green"),
    ("menu_back", "light gray", "dark magenta"),
    ("progress", "white", "default"),
    ("pg_done", "white", "dark green", "bold"),
    ("log_line", "light cyan", "default"),
    ("form_label", "light cyan", "default"),
    ("form_label_disabled", "dark gray", "default"),
    ("form_error", "light red", "default"),
    ("cuda_tag", "light green", "dark blue", "bold"),
    ("cpu_tag", "light gray", "dark blue"),
    ("focus_border", "light cyan", "default", "bold"),
    ("focus_title", "light cyan", "default", "bold"),
    ("dim_border", "dark gray", "default"),
    ("dim_title", "dark gray", "default", "bold"),
]

# ------------------------------------------------------------------ #
#  Dialog attribute remap
# ------------------------------------------------------------------ #

DIALOG_BG = "dark gray"

DIALOG_REMAP: dict = {None: "dialog"}
_dialog_variants = []
for _name, _fg, _bg, *_rest in PALETTE:
    if _bg != "default":
        continue
    _twin = f"{_name}@dlg"
    _twin_fg = "light gray" if _fg == "dark gray" else _fg
    _dialog_variants.append((_twin, _twin_fg, DIALOG_BG, *_rest))
    DIALOG_REMAP[_name] = _twin

PALETTE.extend(_dialog_variants)


# ------------------------------------------------------------------ #
#  Color depth
# ------------------------------------------------------------------ #

TRUECOLOR = 2 ** 24

_TRUECOLOR_TERM_PROGRAMS = {
    "vscode", "iTerm.app", "WezTerm", "ghostty", "Hyper", "rio",
}

_COLOR_NAMES = {
    "truecolor": TRUECOLOR,
    "24bit": TRUECOLOR,
    "24-bit": TRUECOLOR,
    "256": 256,
    "16": 16,
    "1": 1,
}


def detect_color_depth(env: Mapping[str, str] | None = None) -> int:
    env = os.environ if env is None else env

    override = env.get("RAPMAT_COLORS", "").strip().lower()
    if override in _COLOR_NAMES:
        return _COLOR_NAMES[override]

    colorterm = env.get("COLORTERM", "").lower()
    if "truecolor" in colorterm or "24bit" in colorterm:
        return TRUECOLOR
    if env.get("WT_SESSION") or env.get("KITTY_WINDOW_ID"):
        return TRUECOLOR
    if env.get("ALACRITTY_WINDOW_ID"):
        return TRUECOLOR
    if env.get("TERM_PROGRAM") in _TRUECOLOR_TERM_PROGRAMS:
        return TRUECOLOR

    term = env.get("TERM", "")
    if "direct" in term or "truecolor" in term:
        return TRUECOLOR
    if "256color" in term:
        return 256
    if not term and sys.platform == "win32":
        return TRUECOLOR
    return 16


def color_depth_label(depth: int) -> str:
    if depth >= TRUECOLOR:
        return "truecolor"
    if depth > 16:
        return f"{depth} colours (approximated)"
    return f"{depth} colours"


def apply_color_depth(screen, depth: int | None = None) -> int:
    if depth is None:
        depth = detect_color_depth()

    setter = getattr(screen, "set_terminal_properties", None)
    if setter is None:
        return 16

    for candidate in (depth, 256, 16):
        if candidate > depth:
            continue
        try:
            setter(colors=candidate)
            return candidate
        except Exception:
            continue
    return 16
