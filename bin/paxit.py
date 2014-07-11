#!/usr/bin/env python
from pax import pax

if __name__ == '__main__':
    pax.Processor(input='DumbExample.DumbExampleInput',
                  transform='DumbExample.DumbExampleTransform',
                  output='DumbExample.DumbExampleOutput')

