import os
import logging
import logging.config


config_dict = {
    "version": 1,
    "handlers": {
        "fileHandler": {
            "class": "logging.FileHandler",
            "formatter": "myFormatter",
            "filename": "train_log.log",
        },
        "consoleHandler": {
            "class": "logging.StreamHandler",
            "formatter": "myFormatter",
        },
    },
    "loggers": {
        "wgan_gp_logger": {
            "handlers": ["fileHandler", "consoleHandler"],
            "level": "INFO",
        },
    },
    "formatters": {
        "myFormatter": {
            "format": "%(asctime)s : [%(name)s] : [%(levelname)s] - %(message)s"
        }
    }
}

if os.path.exists("./train_log.log"):
    os.remove("./train_log.log")
logging.config.dictConfig(config_dict)
wgan_gp_logger = logging.getLogger("wgan_gp_logger")
