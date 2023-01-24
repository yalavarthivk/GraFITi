r"""Tests for tsdm."""

# import logging
# __logger__ = logging.getLogger(__name__)
#
# TEST_LEVELV_NUM = 9
# logging.addLevelName(TEST_LEVELV_NUM, "TEST")
#
# GREY = "\x1b[38;21m"
# YELLOW = "\x1b[33;21m"
# RED = "\x1b[31;21m"
# BOLD_RED = "\x1b[31;1m"
# RESET = "\x1b[0m"
# GREEN = "\x1b[1;32m"
#
#
# class CustomFormatter(logging.Formatter):
#
#     format = (
#         "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
#     )
#
#     FORMATS = {
#         logging.DEBUG: GREY + format + RESET,
#         logging.INFO: GREY + format + RESET,
#         logging.WARNING: YELLOW + format + RESET,
#         logging.ERROR: RED + format + RESET,
#         logging.CRITICAL: BOLD_RED + format + RESET,
#     }
#
#     def format(self, record):
#         log_fmt = self.FORMATS.get(record.levelno)
#         formatter = logging.Formatter(log_fmt)
#         return formatter.format(record)
#
#
# def failed(self, message, *args, **kws):
#     # Yes, logger takes its '*args' as 'args'.
#     message = BOLD_RED + "✘" + message + "FAILED ✘" + RESET
#     self._log(TEST_LEVELV_NUM, message, args, **kws)
#
#
# def passed(self, message, *args, **kws):
#     # Yes, logger takes its '*args' as 'args'.
#     message = GREEN + "✔" + message + " PASSED ✔" + RESET
#     self._log(TEST_LEVELV_NUM, message, args, **kws)
#
#
# logging.Logger.failed = failed
# logging.Logger.passed = passed
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
#
# ch.setFormatter(CustomFormatter())
# __logger__.addHandler(ch)
# __logger__.debug("debug message")
# __logger__.info("info message")
# __logger__.warning("warning message")
# __logger__.error("error message")
# __logger__.critical("critical message")
