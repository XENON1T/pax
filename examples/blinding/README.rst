Blinding demonstration
======================

This demonstrates pax's proposed blinding module with a blinding cut that simply blinds all even events. In each step of the unblinding process, this readme will first explain what should happen in real life, then explain the example from this directory which represents that step.

**Step 1**: Pax runs on the dark matter data with the blinding cut plugin just before the output plugin. Events that pass a blinding cut are skipped (so their processed data is not output) and their event numbers are stored in a text file.

    paxer --config_path process_with_blinding_cut.ini

This runs pax on the first 10 events of the example AmBe data file (eventually to be the MongoDB data set containing dark matter data). It produces a file `processed_nonblind` with data from the nonblinded (odd) events, as well as a file `blind_events.txt` with the numbers of the blind events (0, 2, 4, 8). If you'd like waveforms instead of processed data, add `--plot` or even `--plot_to_dir plots_unblinded`.

**Step 2**: Pax is run again to split the raw data. One pax reads just the blind events and writes them to a file encrypted with a public key in the config. Another pax reads all other events and makes an unprotected raw data file.

    paxer --config_path write_blind_data.ini
    paxer --config_path write_nonblind_data.ini
    
These will produce the `blind_raw_data` and `nonblind_raw_data` folders. The nonblind data can be read: try e.g. `paxer --config ZippedBSON --input nonblind_raw_data --plot`. The blinded data cannot be read: `paxer --config ZippedBSON --input blind_raw_data --plot` will crash (`OSError: Not a gzipped file`). It has been encrypted with the public key `blinding_example_public_key.pklz` in `pax/data`.

**Step 3**: The raw data should now be removed from Mongo. Analysis can commence on the nonblind events: the raw data can be backed up, sent around, waveforms can be plotted, etc.

This you'll have to imagine in your head.

**Step 4**: When unblinding arrives, the blinded raw data is decrypted using a private key held by a trusted member. This should be processed by pax as usual (with the same configuration of course).

    paxer --config_path unblind.ini
    
This produces a file `processed_unblinded` with data from the unblinded events. If you'd like waveforms instead of processed data, add `--plot` or even `--plot_to_dir plots_unblinded`. The blinded events are read using the private key `blinding_example_private_key.pklz` in `pax/data`. Of course we would not normally put the private key in pax! 