"""Define unit system for pax (i.e., seconds, etc.)

This sets up variables for the various unit abbreviations, ensuring we always
have a 'consistent' unit system.
You can change base_units without breaking the consistency of any physical
relations

Please change comments in this code if you change any units. Better yet, don't
change any units!

In contrast to the existing units.py, explicit_units.py does not generate the attributes via the vars() command.
Instead all necessary units are defined explicitly. With this code completion within IDEs will work and no more
warnings are shown.
Also probably never used units like nHz can be considered to be missed out.
"""

# From physics.nist.gov, January 2015
electron_charge_SI = 1.602176565 * 10 ** (-19)
boltzmannConstant_SI = 1.3806488 * 10 ** (-23)

m = 10 ** 2  # distances in cm
s = 10 ** 9  # times in ns
eV = 1  # energies in eV
C = 1  # Charge in number of electrons (so voltage will be in Volts)
K = 1  # Temperature in Kelvins

# derived units
Hz = 1 / s
J = eV / electron_charge_SI
kg = J * s ** 2 / m ** 2
V = J / C
A = C / s
N = J / m
Pa = N / m ** 2
bar = 10 ** 5 * Pa
Ohm = V / A

###### 10 ^ -3 base units
mm = 10 ** (-3) * m
ms = 10 ** (-3) * s
mK = 10 ** (-3) * K
mC = 10 ** (-3) * C

mHz = 10 ** (-3) * Hz
mJ = 10 ** (-3) * J
g = 10 ** (-3) * kg
mV = 10 ** (-3) * V
mA = 10 ** (-3) * A
mN = 10 ** (-3) * N
mPa = 10 ** (-3) * Pa
mbar = 10 ** (-3) * bar
mOhm = 10 ** (-3) * Ohm

###### 10 ^ -6 base units
um = 10 ** (-6) * m
us = 10 ** (-6) * s
uK = 10 ** (-6) * K
uC = 10 ** (-6) * C

uHz = 10 ** (-6) * Hz
uJ = 10 ** (-6) * J
mg = 10 ** (-6) * kg
uV = 10 ** (-6) * V
uA = 10 ** (-6) * A
uN = 10 ** (-6) * N
uPa = 10 ** (-6) * Pa
ubar = 10 ** (-6) * bar
uOhm = 10 ** (-6) * Ohm

###### 10 ^ -9 base units
nm = 10 ** (-9) * m
ns = 10 ** (-9) * s
nK = 10 ** (-9) * K
nC = 10 ** (-9) * C

nJ = 10 ** (-9) * J
ug = 10 ** (-9) * kg
uV = 10 ** (-9) * V

####### 10 ^ 3 base units
km = 10 ** 3 * m
kJ = 10 ** 3 * J
kV = 10 ** 3 * V
kHz = 10 ** 3 * Hz
kA = 10 ** 3 * A
kN = 10 ** 3 * N
kOhm = 10 ** 3 * Ohm
kPa = 10 ** 3 * Pa
keV = 10 ** 3 * eV

####### 10 ^ 6 base units
MJ = 10 ** 6 * J
MV = 10 ** 6 * V
MHz = 10 ** 6 * Hz
MN = 10 ** 6 * N
MOhm = 10 ** 6 * Ohm
MPa = 10 ** 6 * Pa
MeV = 10 ** 6 * eV

####### 10 ^ 9 base units
GJ = 10 ** 9 * J
GHz = 10 ** 9 * Hz
GPa = 10 ** 9 * Pa

####### other units
cm = 10 ** (-2) * m
# Townsend (unit for reduced electric field)
Td = 10 ** (-17) * V / cm ** 2  # noqa

electron_charge = electron_charge_SI * C
boltzmannConstant = boltzmannConstant_SI * J / K
