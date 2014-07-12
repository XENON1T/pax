from configglue import schema


class XedReader(schema.Schema):
    gain = schema.IntOption(default=2*10**6)
    digitizer_resistor   = schema.IntOption(default=50)
    digitizer_amplification = schema.IntOption(default=10)