from distutils.core import setup, Extension


from distutils import sysconfig
import platform

if platform.system() != 'Windows':  # When compilinig con visual no -g is added to params
	cflags = sysconfig.get_config_var('CFLAGS')
	opt = sysconfig.get_config_var('OPT')
	print(cflags)
	print(opt)
	#sysconfig._config_vars['CFLAGS'] = cflags.replace(' -g ', ' ')
	#sysconfig._config_vars['OPT'] = opt.replace(' -g ', ' ')

if platform.system() == 'Linux':  # In macos there seems not to be -g in LDSHARED
	ldshared = sysconfig.get_config_var('LDSHARED')
	print(ldshared)
	#sysconfig._config_vars['LDSHARED'] = ldshared.replace(' -g ', ' ')


setup (
		name = 'detexpy',
		version = '1.0',
		description = 'Detex Python port. Software for removing Latex commands from .tex files to produce .txt files.',
		author = 'Daniel Trinkle (Python port by Tom Crossland)',
		ext_modules = [Extension('detexpy', sources = ['detexpysource.c'])]
	)
