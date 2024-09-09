"""Check the compatibility of units."""

def compatible(unit1, unit2):
    """True iff quantities in the given units can be interchanged
    for some multiplier.
    """
    #if hasattr(unit1,'canonical') and hasattr(unit2,'canonical'):
        #return unit1.canonical() == unit2.canonical()
    if hasattr(unit1,'immutable') and hasattr(unit2,'immutable'):
        return unit1.immutable() == unit2.immutable()
    else:
        return False

def within_epsilon(quantity1, quantity2, epsilon=10**-9):
    """True iff the given quantities are close to each other within reason."""
    return abs(quantity1 - quantity2).num < epsilon
