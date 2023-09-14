"""Custom logger implementation to manage and subsequently write log data for individual simulation runs.
"""
from pathlib import Path
import logging, logging.handlers
from itertools import repeat

import numpy as np


class EbsLogger:

    """Custom logger class.

    Attributes:
        data_header (list<str>): Parameter names to be logged.
        file_path (path): Path of the log file
        logger (logging handler): Handler of the logger
        seperator (str): Seperator used in the output file to delimit data within a row.
    """

    file_path = None
    data_header = None
    seperator = ","
    logger = None

    def __init__(
        self,
        data_header,
        file_name="Param_Log.txt",
        folder_name="Logs",
        logger_name="Default",
        mirror=False,
    ):
        """Ctor

        Args:
            data_header (list<str>): Parameter names to be logged.
            file_name (str, optional): Name of the log file.
            folder_name (str, optional): Name of the output folder
            logger_name (str, optional): Name of the logger instance
            mirror (bool, optional): Whether or not to write log data to the command line output.
        """
        self.data_header = data_header  ## list of parameters that are contained in data (column header)
        self.create_logger(
            logger_name=logger_name,
            log_filename=file_name,
            log_folder=folder_name,
            mirror_on_console=mirror,
        )
        self.logger.info(self.seperator.join(data_header))

    def create_logger(
        self, logger_name, log_filename, log_folder="Logs", mirror_on_console=False
    ):
        """Instantiate and pre-configure the logger instance.

        Args:
            logger_name (str): Name of the logger instance
            log_filename (str): Name of the log file.
            log_folder (str, optional):  Name of the output folder
            mirror_on_console (bool, optional): Whether or not to write log data to the command line output.
        """
        Path(str(log_folder)).mkdir(parents=True, exist_ok=True)
        self.file_path = str(log_folder) + "/" + log_filename

        formatter = logging.Formatter("%(message)s")  # "%(asctime)s, %(message)s"
        file_handler = logging.FileHandler(self.file_path)
        file_handler.setFormatter(formatter)

        nr_records_before_writing = 1024
        memory_handler_file = logging.handlers.MemoryHandler(
            capacity=nr_records_before_writing, target=file_handler
        )

        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.addHandler(memory_handler_file)

        if mirror_on_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            memory_handler_console = logging.handlers.MemoryHandler(
                capacity=nr_records_before_writing, target=console_handler
            )
            logger.addHandler(memory_handler_console)

        self.logger = logger

    def convert_data(self, data, decimals=None):
        """Transform plain data collected in a simulation run into a relation data table.

        Args:
            data (dict): Logged simulation data that should be written as an output
            decimals (None, optional): Number of decimals for numeric data (rounding)

        Returns:
            list<*>: Data row.

        Raises:
            TypeError: Raised if raw data is not in the correct format.
            ValueError: Raised if an error during logging occured which corrupted the raw data.
        """
        if not isinstance(data, dict):
            raise TypeError(
                f"Data must be of type {type(dict())} but is of type {type(data)}."
            )

        if set(data.keys()) - set(self.data_header):
            raise ValueError(
                f"Data contains invalid payload. Configured data to log (header) might not contain {str(set(data.keys())- set(self.data_header))}"
            )

        if not all([len(v) == len(data[self.data_header[0]]) for k, v in data.items()]):
            raise ValueError(
                f"Some entries are missing in the provided data - can not convert it for logging."
            )

        row_data = np.array(
            list(data.values())
        ).transpose()  # , dtype=object).transpose()

        if decimals:
            return [
                self.seperator.join(
                    map(self.round_to_string, row.tolist(), repeat(int(decimals)))
                )
                for row in row_data
            ]

        else:
            return [self.seperator.join(map(str, row.tolist())) for row in row_data]

    def round_to_string(self, x, decimals=2):
        """Converts numeric data to a suitable output resolution and format

        Args:
            x (*): Raw datapoint
            decimals (int, optional): Number of decimals for numeric data (rounding)

        Returns:
            str: Converted datapoint
        """
        return str(round(x, decimals)) if isinstance(x, float) else str(x)

    def write(self, data, decimals=None):
        """Persist simulation data.

        Args:
            data (list<*>): Data row.
            decimals (None, optional): Number of decimals for numeric data (rounding)
        """
        log_messages = self.convert_data(data, decimals)

        for message in log_messages:
            self.logger.info(message)
