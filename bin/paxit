#!/usr/bin/env python
from pax import pax

# Random notes: expand on filter for small and large S2
# If plugins define docstring, we can load this in the docs
# Add issues about dependencies
# Explain what pruning is

config_overload = """
[pax]
input = 'MongoDB.MongoDBInput'
my_extra_transforms = ["PosSimple.PosRecWeightedSum"]
output = ["Plotting.PlottingWaveform"]

[MongoDB.MongoDBInput]
collection = "dataset"
address = "145.102.135.218:27017"
"""

if __name__ == '__main__':
    pax.processor(config_overload)