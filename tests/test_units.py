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
        electron_charge_si = 1.602176565 * 10 ** (-19)
        # boltzmannconstant_si = 1.3806488 * 10 ** (-23)

        base_units = {
            'm': 10 ** 2,  # distances in cm
            's': 10 ** 9,  # times in ns.
            # Note: sample size must be an integer multiple of unit time to use
            # our datastructure.  pax is used on digitizers faster than XE100.
            'eV': 1,  # energies in eV
            # Charge in number of electrons (so voltage will be in Volts)
            'C': 1 / electron_charge_si,
            'K': 1,  # Temperature in Kelvins
        }

        base_units['Hz'] = 1 / base_units['s']
        base_units['J'] = base_units['eV'] / electron_charge_si
        base_units['g'] = 1e-3 * base_units['J'] * base_units['s'] ** 2 / base_units['m'] ** 2
        base_units['V'] = base_units['J'] / base_units['C']
        base_units['A'] = base_units['C'] / base_units['s']
        base_units['N'] = base_units['J'] / base_units['m']
        base_units['Pa'] = base_units['N'] / base_units['m'] ** 2
        base_units['bar'] = 10 ** 5 * base_units['Pa']
        base_units['Ohm'] = base_units['V'] / base_units['A']

        # Make variables for ns, uHz, kOhm, etc.
        prefixes = {'': 0, 'n': -9, 'u': -6, 'm': -3, 'k': 3, 'M': 6, 'G': 9}
        for (name, value) in list(base_units.items()):
            for (p_name, p_factor) in list(prefixes.items()):
                unit_name = p_name + name
                a = getattr(units, unit_name)
                b = float(10 ** p_factor * value)
                diff = (a - b)

                self.assertAlmostEqual(a,
                                       b,
                                       delta=1e-5 * b,  # How big can diff be?
                                       msg='%s is wrong by %f' % (unit_name,
                                                                  diff))
        base_units['cm'] = base_units['m'] / 100

        self.assertAlmostEqual(units.cm, base_units['cm'])
if __name__ == '__main__':
    unittest.main()
