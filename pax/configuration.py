from confiture.schema.containers import Section, Value, List
from confiture.schema.types import String, Float, Integer
from pax import units

BAD_PMTS = [1, 2, 145, 148, 157, 171, 177]


class PaxSchema(Section):

    """Schema for PAX

    TODO: structure this a bit more as it gets more complicated with
    subsections.
    """

    # Sum waveform
    gain = Value(Float(), default=2 * 10 ** 6)
    digitizer_resistor = Value(Float(), default=50 * units.Ohm)
    digitizer_amplification = Value(Float(), default=10)

    # Digitizer info
    digitzer_bits = 14
    dt = 10 * units.ns
    digitizer_baseline = Value(Float(), default=(2 ** digitzer_bits - 1))
    digitizer_V_resolution = Value(
        Float(), default=(2.25 * units.V / 2 ** (digitzer_bits)))
    digitizer_t_resolution = Value(Float(), default=(dt))

    # PMTs
    excluded = List(Integer(), default=BAD_PMTS)
    top = List(
        Integer(), default=[x for x in range(1, 98) if x not in BAD_PMTS])
    bottom = List(
        Integer(), default=[x for x in range(99, 178) if x not in BAD_PMTS])
    veto = List(
        Integer(), default=[x for x in range(179, 242) if x not in BAD_PMTS])

    # Peak finding
    threshold = Value(Float(), default=62.415)
    boundary_to_height_ratio = Value(Float(), default=0.005)
    min_length = Value(Integer(), default=int(0.6 * units.us / dt))
    test_before = Value(Integer(), default=int(0.21 * units.us / dt))
    test_after = Value(Integer(), default=int(0.21 * units.us / dt))
    before_to_height_ratio_max = Value(Float(), default=0.05)
    after_to_height_ratio_max = Value(Float(), default=0.05)

    # Mongo
    # TODO (tunnell): Trouble with ipaddr module in Py3? Can't use IPAddress
    #                 type.
    # TODO (tunnell): Subsection this!

    database = Value(String(), default="output")
    collection = Value(String(), default="dataset",
                       argparse_names=['--collection'],
                       argparse_help='MongoDB collection to use.')

    mongodb_address = Value(String(), default="127.0.0.1:27017",
                            argparse_names=['--mongo'],
                            argparse_help='MongoDB IP address (e.g., 127.0.0.1'
                                          ' or, for port, 127.0.0.1:27017).')

    # Internal
    plugin_paths = List(String(), argparse_names=['--pluginpaths'],
                        argparse_help='Extra paths to search for plugins.',
                        default=[])
