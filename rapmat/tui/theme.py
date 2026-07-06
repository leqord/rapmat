"""Shared urwid palette.
"""

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
