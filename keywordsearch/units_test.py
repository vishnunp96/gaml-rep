if __name__ == '__main__':

	from gaml.units import unit,named_unit
	from gaml.units.predefined import define_units
	#from gaml.units.compatibility import compatible
	define_units()
	named_unit('sec',['s'],[])

	hubble_unit = unit('km') / unit('s') / unit('Mpc')

	print('km:',unit('km').__repr__())
	print()
	print('Mpc:',unit('Mpc').__repr__())
	print()
	print('1/Mpc:',(1/unit('Mpc')).__repr__())
	print()
	print('Mpc**-1:',(unit('Mpc')**-1).__repr__())
	print()
	print('Mpc**-1 * km:',(unit('Mpc')**-1 * unit('km')).__repr__())
	print()
	print('km*Mpc**-1:',(unit('km') * unit('Mpc')**-1).__repr__())
	print()
