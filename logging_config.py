import logging
from logging.handlers import RotatingFileHandler
import os
import configparser

# Defaults
DEFAULT_LOG_FILE = os.path.join(os.path.dirname(__file__), 'chatpaper.log')
DEFAULT_LEVEL = 'INFO'


def _resolve_config():
    """Resolve logging configuration from (in order):
    1. Environment variables CHATPAPER_LOG_LEVEL, CHATPAPER_LOG_FILE
    2. apikey.ini [Logging] section
    3. Defaults
    Returns (level_name, log_file)
    """
    level = os.environ.get('CHATPAPER_LOG_LEVEL')
    log_file = os.environ.get('CHATPAPER_LOG_FILE')

    # Try reading apikey.ini if env vars not set
    cfg_path = os.path.join(os.path.dirname(__file__), 'apikey.ini')
    if (not level or not log_file) and os.path.exists(cfg_path):
        cfg = configparser.ConfigParser()
        try:
            cfg.read(cfg_path, encoding='utf-8')
            if cfg.has_section('Logging'):
                if not level and cfg.has_option('Logging', 'level'):
                    level = cfg.get('Logging', 'level')
                if not log_file and cfg.has_option('Logging', 'file'):
                    log_file = cfg.get('Logging', 'file')
        except Exception:
            # ignore and fallback to defaults
            pass

    if not level:
        level = DEFAULT_LEVEL
    if not log_file:
        log_file = DEFAULT_LOG_FILE

    return level.upper(), os.path.abspath(log_file)


# Configure root logger only once
root = logging.getLogger()
if not any(getattr(h, 'name', None) == 'chatpaper_file_handler' for h in root.handlers):
    level_name, LOG_FILE = _resolve_config()
    try:
        level = getattr(logging, level_name)
    except Exception:
        level = logging.INFO
    root.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    ch.name = 'chatpaper_console'
    root.addHandler(ch)

    # Rotating file handler
    fh = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5, encoding='utf-8')
    fh.setLevel(level)
    fh.setFormatter(formatter)
    fh.name = 'chatpaper_file_handler'
    root.addHandler(fh)


def get_logger(name=None):
    return logging.getLogger(name)
