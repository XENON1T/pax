
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


class LocalPaxCrashed(PaxException):
    """Can be raised in the host process if one of the local paxes crashes"""
    pass


class RemotePaxChrash(PaxException):
    """Can be raised in the host process if one of the remote paxes in the same chain crashes"""
    pass
