"""Half-block canvas tests."""

import numpy as np
import pytest
import urwid

from rapmat.tui.widgets.halfblock import (
    GLYPH,
    AttrSpecCache,
    HalfBlockCanvas,
    frame_to_canvas,
)


def _fb(rows_of_colors):
    return np.array(rows_of_colors, dtype=np.uint8)


def _uniform(h, w, color=(10, 20, 30)):
    return np.tile(np.array(color, dtype=np.uint8), (h, w, 1))


def _cache():
    return AttrSpecCache()


# ------------------------------------------------------------------ #
#  frame_to_canvas
# ------------------------------------------------------------------ #


def test_attr_run_lengths_are_bytes():
    canvas = frame_to_canvas(_uniform(4, 4), _cache())
    assert canvas.cols() == 4
    assert canvas.rows() == 2
    for row in canvas.content():
        entries = list(row)
        assert len(entries) == 1
        assert entries[0][2] == GLYPH.encode("utf-8") * 4


def test_top_and_bottom_pixels_become_fg_and_bg():
    fb = _fb([[[255, 0, 0]] * 3, [[0, 0, 255]] * 3])
    canvas = frame_to_canvas(fb, _cache())
    spec = list(canvas.content())[0][0][0]
    assert spec.get_rgb_values() == (255, 0, 0, 0, 0, 255)


def test_run_length_compression():
    row_top = [[255, 0, 0]] * 4 + [[0, 255, 0]] * 4
    row_bot = [[0, 0, 0]] * 8
    canvas = frame_to_canvas(_fb([row_top, row_bot]), _cache())
    assert len(list(list(canvas.content())[0])) == 2


def test_every_cell_distinct_gives_one_run_each():
    top = [[i * 8, 0, 0] for i in range(8)]
    canvas = frame_to_canvas(_fb([top, [[0, 0, 0]] * 8]), _cache())
    assert len(list(list(canvas.content())[0])) == 8


def test_odd_pixel_height_is_truncated():
    assert frame_to_canvas(_uniform(5, 4), _cache()).rows() == 2


def test_zero_height_framebuffer():
    assert frame_to_canvas(_uniform(1, 4), _cache()).rows() == 0


def test_canvas_matches_framebuffer_width():
    for w in (1, 7, 80):
        assert frame_to_canvas(_uniform(6, w), _cache()).cols() == w


# ------------------------------------------------------------------ #
#  AttrSpecCache
# ------------------------------------------------------------------ #


def test_cache_returns_identical_instances():
    cache = _cache()
    assert cache[0x123456789A] is cache[0x123456789A]
    assert len(cache) == 1


def test_cache_decodes_key_into_fg_and_bg():
    spec = _cache()[(0xFF8800 << 24) | 0x001122]
    assert spec.get_rgb_values() == (255, 136, 0, 0, 17, 34)


def test_cache_clears_on_overflow():
    cache = AttrSpecCache(maxsize=4)
    for k in range(5):
        assert cache[k] is not None
    assert len(cache) <= 4


def test_set_colors_clears_and_switches_depth():
    cache = _cache()
    cache[0x010203040506]
    cache.set_colors(256)
    assert len(cache) == 0
    assert cache[0x010203040506].colors == 256


def test_set_colors_to_same_value_keeps_cache():
    cache = _cache()
    cache[1]
    cache.set_colors(2 ** 24)
    assert len(cache) == 1


def test_sixteen_color_depth_rejects_true_color():
    with pytest.raises(urwid.AttrSpecError):
        AttrSpecCache(colors=16)[0xFF000000]


# ------------------------------------------------------------------ #
#  HalfBlockCanvas
# ------------------------------------------------------------------ #


def _widget(source=None):
    return HalfBlockCanvas(source or (lambda w, h: _uniform(h, w)))


def test_widget_is_box_and_not_selectable():
    widget = _widget()
    assert urwid.BOX in widget.sizing()
    assert widget.selectable() is False


@pytest.mark.parametrize("size", [(80, 24), (121, 13), (20, 6), (200, 60)])
def test_widget_render_matches_size(size):
    canvas = _widget().render(size)
    assert (canvas.cols(), canvas.rows()) == size


@pytest.mark.parametrize("size", [(10, 3), (19, 20), (30, 5)])
def test_widget_too_small_falls_back(size):
    canvas = _widget().render(size)
    assert (canvas.cols(), canvas.rows()) == size
    assert "too small" in _text_of(canvas)


@pytest.mark.parametrize("size", [(4, 2), (1, 1), (2, 8), (8, 1)])
def test_widget_renders_at_sizes_too_narrow_for_the_message(size):
    canvas = _widget().render(size)
    assert (canvas.cols(), canvas.rows()) == size


def test_widget_zero_size():
    canvas = _widget().render((0, 0))
    assert (canvas.cols(), canvas.rows()) == (0, 0)


def test_widget_survives_source_exception():
    def boom(w, h):
        raise RuntimeError("kaboom")

    canvas = _widget(boom).render((80, 24))
    assert "Render failed" in _text_of(canvas)
    assert "kaboom" in _text_of(canvas)


def test_widget_handles_none_frame():
    assert "Nothing to display" in _text_of(_widget(lambda w, h: None).render((80, 24)))


def test_widget_handles_wrong_shaped_frame():
    canvas = _widget(lambda w, h: _uniform(4, 4)).render((80, 24))
    assert "Nothing to display" in _text_of(canvas)


def test_widget_message_overrides_rendering():
    widget = _widget()
    widget.set_message("no colours here")
    assert "no colours here" in _text_of(widget.render((80, 24)))
    widget.set_message("")
    assert "no colours here" not in _text_of(widget.render((80, 24)))


def test_widget_records_render_time():
    widget = _widget()
    widget.render((80, 24))
    assert widget.last_render_seconds > 0


def test_keys_pass_through_a_non_selectable_body():
    frame = urwid.Frame(body=_widget())
    for key in ("right", "h", "+", " ", "n"):
        assert frame.keypress((80, 24), key) == key


def _text_of(canvas) -> str:
    return "\n".join(
        b"".join(part[2] for part in row).decode("utf-8", "replace")
        for row in canvas.content()
    )
