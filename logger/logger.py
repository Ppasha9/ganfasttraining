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
        "loggers": {
            "wgan_gp_logger": {
                "handlers": ["fileHandler"],
                "level": "INFO",
            },
        },
        "formatters": {
            "myFormatter": {
                "format": "%(asctime)s : [%(name)s] : [%(levelname)s] - %(message)s"
            }
        }
    }
}

logging.config.dictConfig(config_dict)
wgan_gp_logger = logging.getLogger("wgan_gp_logger")
