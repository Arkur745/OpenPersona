import logging
import sys

def configure_logging(level: int = logging.INFO):
    """
    Sets up the global logging configuration for the MemEngine system.
    Redirects stdout logging to structured handlers.
    """
    logger = logging.getLogger("memengine")
    logger.setLevel(level)
    
    # Remove existing handlers to avoid double logging
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

def get_logger(subsystem: str) -> logging.Logger:
    """
    Returns a named logger for a given cognitive subsystem.
    Subsystems should follow the standard hierarchy (e.g. 'decay', 'activation', 'planner').
    """
    return logging.getLogger(f"memengine.{subsystem}")

# Default configuration upon initial import
configure_logging()
