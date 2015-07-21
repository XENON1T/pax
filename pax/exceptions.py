
class PulseBeyondEventError(Exception):
    pass


class OutputFileAlreadyExistsError(Exception):
    pass


class CantReadBlindedEvent(Exception):
    """Thrown by plugins that support blinded events when they don't have a key to decrypt them.
    FolderIO catches and ignores this when iterating over all events in a file,
    but if you specifically asked for an event that throws this, it will halt pax.
    """
    pass
