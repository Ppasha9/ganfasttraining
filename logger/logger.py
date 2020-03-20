import os
import logging
import logging.config


config_dict = {
    "version": 1,
    "handlers": {
        "wgan_gp_file_handler": {
            "class": "logging.FileHandler",
            "formatter": "my_formatter",
            "filename": "train_wgan_gp_log.log",
        },
        "swae_file_handler": {
            "class": "logging.FileHandler",
            "formatter": "my_formatter",
            "filename": "train_swae_log.log",
        },
        "console_handler": {
            "class": "logging.StreamHandler",
            "formatter": "my_formatter",
        },
    },
    "loggers": {
        "wgan_gp_logger": {
            "handlers": ["wgan_gp_file_handler", "console_handler"],
            "level": "INFO",
        },
        "swae_logger": {
            "handlers": ["swae_file_handler", "console_handler"],
            "level": "INFO",
        },
    },
    "formatters": {
        "my_formatter": {
            "format": "%(asctime)s : [%(name)s] : [%(levelname)s] - %(message)s"
        }
    }
}

if os.path.exists("./train_log.log"):
    os.remove("./train_log.log")
logging.config.dictConfig(config_dict)
wgan_gp_logger = logging.getLogger("wgan_gp_logger")
swae_logger = logging.getLogger("swae_logger")
