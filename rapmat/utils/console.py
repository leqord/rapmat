import logging

from rapmat.config import APP_DATA_DIR

_LOG_DIR = APP_DATA_DIR / "logs"
_LOG_FILE = _LOG_DIR / "background.log"

_THIRD_PARTY_LEVEL = logging.WARNING

_configured = False
_file_handler: logging.Handler | None = None


def _build_file_handler() -> logging.Handler:
    global _file_handler
    if _file_handler is None:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        _file_handler = handler
    return _file_handler


def configure_logging() -> None:
    """Route all logging to ``background.log`` and keep it off the TUI.
    """
    global _configured
    if _configured:
        return

    handler = _build_file_handler()

    rapmat_logger = logging.getLogger("rapmat")
    if handler not in rapmat_logger.handlers:
        rapmat_logger.addHandler(handler)
    rapmat_logger.setLevel(logging.DEBUG)
    rapmat_logger.propagate = False

    root_logger = logging.getLogger()
    if handler not in root_logger.handlers:
        root_logger.addHandler(handler)
    root_logger.setLevel(_THIRD_PARTY_LEVEL)

    _configured = True


def get_logger(name: str = "rapmat") -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
