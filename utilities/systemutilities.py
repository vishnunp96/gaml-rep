import subprocess

def console(cmd, timeout=None, input=None):
	## The Pythonic setup may require a .terminate() rather than a .kill()?
	try:
		p = subprocess.Popen(('timeout -k {:d} {:d} '.format(int(timeout*1.5),int(timeout)) if timeout else '') + cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
		out, err = p.communicate(input=input)
		return (p.returncode==0, out, err)
	except (KeyboardInterrupt, SystemExit):
		raise
	except Exception as e:
		return (False, None, e.__class__.__name__ + ' ' + str(e))

def debug(cmd, timeout=None):
	if timeout:
		print('cmd: ' + cmd + ' (timeout=' + str(timeout) + ')')
	else:
		print('cmd:',cmd)

	success, output, error = console(cmd, timeout=timeout)

	print('Succes:',success)
	print('Output:',output)
	print('Error:',error)


if __name__ == '__main__':

	## Test timeout functionality
	debug('sleep 5 && echo done',10)
	debug('sleep 5 && echo done',2)
