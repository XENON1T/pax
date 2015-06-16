============
Event format
============

In this document, we explain what lies within the `pax` event class and how to use it.  This even class is central to how `pax` works but also in how you analyze the resulting data.  All communication within `pax` is done by passing around this event class, therefore it contains all the information you should need.  We also serialize this event class to various outputs that contain all of the information below.  

Please be aware that the event class is a tree, where e.g. there is a list of peak objects associated with the event.  Lastly, this document is automatically updated every release, so please feel free to expand upon the documentation in the source code (see pax/datastructure.py).

-----
Event
-----

.. autoclass:: pax.datastructure.Event
    :members:
    :undoc-members:

----
Peak
----

.. autoclass:: pax.datastructure.Peak
    :members:
    :undoc-members:
   
---------------------
ReconstructedPosition
---------------------

.. autoclass:: pax.datastructure.ReconstructedPosition
    :members:
    :undoc-members:

---
Hit
---

.. autoclass:: pax.datastructure.Hit
    :members:
    :undoc-members:    

---------
Waveforms
---------

.. autoclass:: pax.datastructure.SumWaveform
    :members:
    :undoc-members:
    

