"""Tests for logging configuration.
"""

import io
import logging

import pytest

from rapmat.utils import console


@pytest.fixture
def isolated_logging(tmp_path, monkeypatch):
    """Run ``configure_logging`` with a temp log file with global logging
    state snapshotted and restored."""
    log_dir = tmp_path / "logs"
    log_file = log_dir / "background.log"
    monkeypatch.setattr(console, "_LOG_DIR", log_dir)
    monkeypatch.setattr(console, "_LOG_FILE", log_file)

    monkeypatch.setattr(console, "_configured", False)
    monkeypatch.setattr(console, "_file_handler", None)

    rapmat_logger = logging.getLogger("rapmat")
    root_logger = logging.getLogger()
    saved = {
        "rapmat_handlers": list(rapmat_logger.handlers),
        "rapmat_level": rapmat_logger.level,
        "rapmat_propagate": rapmat_logger.propagate,
        "root_handlers": list(root_logger.handlers),
        "root_level": root_logger.level,
    }
    rapmat_logger.handlers.clear()
    root_logger.handlers.clear()

    try:
        yield log_file
    finally:
        if console._file_handler is not None:
            console._file_handler.close()
        rapmat_logger.handlers[:] = saved["rapmat_handlers"]
        rapmat_logger.setLevel(saved["rapmat_level"])
        rapmat_logger.propagate = saved["rapmat_propagate"]
        root_logger.handlers[:] = saved["root_handlers"]
        root_logger.setLevel(saved["root_level"])


def test_rapmat_logs_do_not_reach_root_stream(isolated_logging):
    """A stderr-like handler on root (as basicConfig adds) must never see
    rapmat records, otherwise they paint over the TUI."""
    console.configure_logging()

    stream = io.StringIO()
    root_stream_handler = logging.StreamHandler(stream)
    logging.getLogger().addHandler(root_stream_handler)
    try:
        console.get_logger("rapmat.task").info("Processing deformed structure 1/2")
    finally:
        logging.getLogger().removeHandler(root_stream_handler)

    assert stream.getvalue() == ""


def test_basicconfig_is_a_noop_after_configure(isolated_logging):
    """Because root already owns a handler, a library calling basicConfig later
    must not be able to install a StreamHandler."""
    console.configure_logging()

    def plain_stream_handlers(logger):
        return [
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]

    root = logging.getLogger()
    before = list(root.handlers)
    streams_before = len(plain_stream_handlers(root))

    logging.basicConfig()

    assert root.handlers == before
    assert len(plain_stream_handlers(root)) == streams_before


def test_logger_hierarchy_is_configured_once(isolated_logging):
    """rapmat owns a single handler and does not propagate."""
    console.configure_logging()
    console.configure_logging() 

    rapmat_logger = logging.getLogger("rapmat")
    assert len(rapmat_logger.handlers) == 1
    assert rapmat_logger.propagate is False
    assert rapmat_logger.level == logging.DEBUG

    child = console.get_logger("rapmat.task")
    assert child.handlers == []


def test_records_are_written_to_file(isolated_logging):
    log_file = isolated_logging
    console.configure_logging()

    console.get_logger("rapmat.task").info("Processing deformed structure 121/240")
    for handler in logging.getLogger("rapmat").handlers:
        handler.flush()

    assert log_file.exists()
    contents = log_file.read_text(encoding="utf-8")
    assert "Processing deformed structure 121/240" in contents
    assert "rapmat.task" in contents
