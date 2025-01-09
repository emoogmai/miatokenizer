import os
import logging.config
import yaml

#First of all we are going to create a folder where log files will live
miaTokLogsFolder = os.path.join(os.path.expanduser("~"), "miatoklogs")
print("logs folder: {}".format(miaTokLogsFolder))
os.makedirs(miaTokLogsFolder, exist_ok=True)

#Load the configuration file and config logger based on setting on this file
with open("configs/logging_config.yml", "rt") as f:
    config = yaml.safe_load(f.read())
    # Configure the logging module with the config file already loaded
    logging.config.dictConfig(config)

# Get a parent logger object for miatokenizer
GMAI_LOGDEF = os.getenv("GMAI_LOGDEF", "development")
miaToklogger = logging.getLogger(GMAI_LOGDEF)

miaToklogger.info("Logging configuration for miatokenizer is done using {} definition !!!".format(GMAI_LOGDEF))
