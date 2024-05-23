from loguru import logger

def setup_logger(log_file="pipeline.log"):
    """Configures the logger and adds a file sink."""
    logger.remove(0)  # Remove default handler
    logger.add(
        log_file,
        level="DEBUG",  # Adjust the logging level as needed
        rotation="5 MB",  # Rotate logs every 5 MB (optional)
        retention="10 days",  # Keep logs for 10 days (optional)
    )

# Initialize the logger (call this once at the beginning of main.py)
setup_logger()
