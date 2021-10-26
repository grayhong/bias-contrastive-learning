import logging
from logging.config import dictConfig


# [%(asctime)s] [%(levelname)s]

def set_logging(logger_name, level, work_dir):
    LOGGING = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                "format": f"%(message)s"
            },
        },
        "handlers": {
            "console": {
                "level": f"{level}",
                "class": "logging.StreamHandler",
                'formatter': 'simple',
            },
            'file': {
                'level': f"{level}",
                'formatter': 'simple',
                'class': 'logging.FileHandler',
                'filename': f'{work_dir if work_dir is not None else "."}/train.log',
                'mode': 'a',
            },
        },
        "loggers": {
            "": {
                "level": f"{level}",
                "handlers": ["console", "file"] if work_dir is not None else ["console"],
            },
        },
    }
    dictConfig(LOGGING)
    logging.info(f"Log level set to: {level}")
