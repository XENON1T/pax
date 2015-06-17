"""
test_pax_configuration
----------------------------------

Tests for `pax.configuration` module.
"""
import unittest

from pax import units


class TestPaxUnits(unittest.TestCase):

    def test_parsing(self):
        self.assertAlmostEqual(units.Ohm, 1.6021765699999998e-10)

        # From physics.nist.gov, January 2015
        electron_charge_SI = 1.602176565 * 10 ** (-19)
        # boltzmannConstant_SI = 1.3806488 * 10 ** (-23)

        base_units = {
            'm': 10 ** 2,  # distances in cm
            's': 10 ** 9,  # times in ns.
            # Note: sample size must be an integer multiple of unit time to use our
            # datastructure!
            'eV': 1,  # energies in eV
            # Charge in number of electrons (so voltage will be in Volts)
            'C': 1 / electron_charge_SI,
            'K': 1,  # Temperature in Kelvins
        }

        base_units['Hz'] = 1 / base_units['s']
        base_units['J'] = base_units['eV'] / electron_charge_SI
        base_units['g'] = 10 ** (-3) * base_units['J'] * base_units['s'] ** 2 / base_units['m'] ** 2
        base_units['V'] = base_units['J'] / base_units['C']
        base_units['A'] = base_units['C'] / base_units['s']
        base_units['N'] = base_units['J'] / base_units['m']
        base_units['Pa'] = base_units['N'] / base_units['m'] ** 2
        base_units['bar'] = 10 ** 5 * base_units['Pa']
        base_units['Ohm'] = base_units['V'] / base_units['A']

        # Make variables for ns, uHz, kOhm, etc.
        prefixes = {'': 0, 'n': -9, 'u': -6, 'm': -3, 'c': -2, 'k': 3, 'M': 6, 'G': 9}
        for (name, value) in list(base_units.items()):
            for (p_name, p_factor) in list(prefixes.items()):
                self.assertEqual(getattr(units, p_name + name), float(10 ** (p_factor) * value))

if __name__ == '__main__':
    unittest.main()
