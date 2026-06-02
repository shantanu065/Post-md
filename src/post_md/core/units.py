"""Internal canonical units: length in Å, time in ps.

GROMACS files are in nm/ps; readers must convert to Å on the way in so the
analysis layer is unit-agnostic. AMBER files are already in Å.
"""

NM_TO_ANGSTROM = 10.0
ANGSTROM_TO_NM = 0.1
