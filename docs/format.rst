============
Event format
============

An event is passed and returned by every module.

* `module`: identifies the digitizer board.  For fake data, this is set to -1.
* `channel`: Channel number.
* `time`: 64-bit unsigned number in units of 10 ns.  This time corresponds to the same time of the data in this document.  Please begin at 1 second or something to prevent trigger trying to save data at negative times.  Do not use floats!
* `data`: An array of 16-bit unsigned numbers, corresponding to the actual data payload.  It is stored as a binary array.


.. code-block:: javascript

    {
        "channel_waveforms": see blah,
        "processed_waveforms": see blah,
        "time": int,
        "data": binary,
    }

Blah blah

.. graphviz::

   digraph foo {
      "event" -> "baz";
   }


Channel waveforms
=================

Fill me.