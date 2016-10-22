
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
