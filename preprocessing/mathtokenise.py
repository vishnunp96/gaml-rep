import re

def tokenise_math(s):
	r = s
	#print('\t',r)
	# Remove empty {}
	r = re.sub('^\{\s*\}',' ',r)
	#print('\t',r)
	# Add spaces around brackets and some math signs
	for t in '{}()[]^_=/<>':
		r = r.replace(t,' '+t+' ')
	#print('\t',r)
	# Add spaces around \-prefixed commands
	r = re.sub('\\\\[^0-9\s\\\\]+',lambda m: ' '+m.group()+' ',r)
	#print('\t',r)
	# Deal with + and - signs
	r = re.sub('-(?=[^0-9])|(?<=\w)-',' - ',r)
	r = re.sub('\+(?=[^0-9])|(?<=\w)\+',' + ',r)
	#print('\t',r)
	# Deal with numbers (digits prefixed with + or - at this point include the sign)
	r = re.sub('(?:\+|-)?[0-9\.]+',lambda m: ' '+m.group()+' ',r)
	#print('\t',r)

	r = re.sub('\s+',' ',r).strip()
	#print('\t',r)
	return r

def regularize_math(s):
	r = s
	#print(r)
	r = re.sub('\s+',' ',r)
	r = re.sub('\\\\(?:quad|,|:|;|!|\ |qquad)',' ',r)
	r = re.sub('\\\\rm{([^{}]+)}',lambda m: m.group(1),r)
	r = re.sub('\\\\rm',' ',r)
	return tokenise_math(r)


