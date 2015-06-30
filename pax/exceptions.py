
class PulseBeyondEventError(Exception):
    pass


class OutputFileAlreadyExistsError(Exception):
    pass


class SkipEvent(Exception):
    """ Raising this exception will cause the processor to skip all further processing for this event, SILENTLY
    I repeat -- SILENTLY!!!
    i.e. don't use this unless you know what you're doing. It exists for an internal purpose (SelectionPlugin).
    """
    pass
