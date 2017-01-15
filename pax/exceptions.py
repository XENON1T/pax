
class PaxException(Exception):
    pass


class PulseBeyondEventError(PaxException):
    pass


class OutputFileAlreadyExistsError(PaxException):
    pass


class CoordinateOutOfRangeException(PaxException):
    pass


class MaybeOldFormatException(PaxException):
    pass


class InvalidConfigurationError(PaxException):
    pass


class TriggerGroupSignals(PaxException):
    pass


class QueueTimeoutException(PaxException):
    pass


class EventBlockHeapSizeExceededException(PaxException):
    pass


class DatabaseConnectivityError(PaxException):
    """A database connectivity error ("Failed to resolve") we often see probably due to a small network hickup.
    """
    pass


class UnknownPropagatedException(Exception):
    """For re-raising an exception of an unknown type in a host process.
    Do NOT subclass PaxException! We don't know where this exception came from.
    """
    pass
