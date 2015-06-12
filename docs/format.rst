============
Event format
============

In this document, we explain the `pax` event class.  This even class is central to how `pax` works but also in how you analyze the resulting data.



All data available for the current event is stored in the event data structure. Input plugins are required to fill the event with a bare amount of raw information. Transform plugins subsequently add their processed variables. 


.. autoclass:: pax.datastructure.Event
    :members:
    :undoc-members:

There are variables within the class that refer to :class:`pax.datastructure.Peak` and :class:`pax.datastructure.ReconstructedPosition` objects.  These have their own class structure.  For example, both S1 and S2 peaks follow the following syntax.

.. autoclass:: pax.datastructure.Peak
    :members:
    :undoc-members:
    
Each reconstruction algorithm creates a :class:`pax.datastructure.ReconstructedPosition` using just the peak information.
    
.. autoclass:: pax.datastructure.ReconstructedPosition
    :members:
    :undoc-members:

.. autoclass:: pax.datastructure.Hit
    :members:
    :undoc-members:    

Waveforms, which are not typically saved, follow the following format.

.. autoclass:: pax.datastructure.SumWaveform
    :members:
    :undoc-members:
    

