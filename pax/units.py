"""Define unit system for pax (i.e., seconds, etc.)

This sets up variables for the various unit abbreviations, ensuring we always
have a 'consistent' unit system.
You can change base_units without breaking the consistency of any physical
relations

Please change comments in this code if you change any units. Better yet, don't
change any units!
"""

# From physics.nist.gov, January 2015
electron_charge_SI = 1.602176565 * 10 ** (-19)
boltzmannConstant_SI = 1.3806488 * 10 ** (-23)

# Here we set our unit system
base_units = {
    'm': 10 ** 2,  # distances in cm
    's': 10 ** 9,  # times in ns.
    # Note: sample size must be an integer multiple of unit time to use our
    # datastructure!
    'eV': 1,       # energies in eV
    # Charge in number of electrons (so voltage will be in Volts)
    'C': 1 / electron_charge_SI,
    'K': 1,        # Temperature in Kelvins
}
# Consequences of this unit system:
#   Current will be in electrons/ns, not A!
#   Resistance in V/(C/s), not Ohm!

# Derive secondary units from the base values - don't change this
base_units['Hz'] = 1 / base_units['s']
base_units['J'] = base_units['eV'] / electron_charge_SI
base_units['g'] = 10**(-3) * base_units['J'] * base_units['s'] ** 2 / base_units['m'] ** 2
base_units['V'] = base_units['J'] / base_units['C']
base_units['A'] = base_units['C'] / base_units['s']
base_units['N'] = base_units['J'] / base_units['m']
base_units['Pa'] = base_units['N'] / base_units['m'] ** 2
base_units['bar'] = 10**5 * base_units['Pa']
base_units['Ohm'] = base_units['V'] / base_units['A']
electron_charge = electron_charge_SI * base_units['C']
boltzmannConstant = boltzmannConstant_SI * base_units['J'] / base_units['K']

# Make variables for ns, uHz, kOhm, etc.
# Unfortunately this won't be recognized as declared variables so that you will get warnings
# that for example unit.ns is not defined. By the use of the command vars() all units come
# into the current name space during runtime, including probably useless units aka GC, nHz, Gs etc.
# We could think of making a hardcoded list with only the useful units as the math won't change.
prefixes = {'': 0, 'n': -9, 'u': -6, 'm': -3, 'c': -2, 'k': 3, 'M': 6, 'G': 9}
for (name, value) in list(base_units.items()):
    for (p_name, p_factor) in list(prefixes.items()):
        # Float makes sure units might work even for poor fellas who forget to
        # from __future__ import division -- not tested though
        vars()[p_name + name] = float(10 ** (p_factor) * value)

# Townsend (unit for reduced electric field)
Td = 10**(-17) * V / cm ** 2    # noqa

