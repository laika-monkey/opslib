def combination(arrs):
	''' generate a cartesian product of input arrays '''
	import itertools
	return list(itertools.product(*arrs))

def intersection(a):
	''' 
	Given a 1D list, return common elements
	a	iter(2), Two iterable objects
	'''
	
	#_set(a).intersection(b)
	x = a.pop()
	y = a.pop()
	
	x = x if hasattr(x, '__iter__') else [x]
	y = y if hasattr(y, '__iter__') else [y]

	inter = list( set(x) & set(y) )
	if len(a) > 0:
		for i in xrange( len(a) ):
			inter = list( set(inter) & set(a[i]) )
	return inter

def mkdir_p(path, mode=0755, **kwargs):
	''' basically mkdir -p '''
	import errno, os
	if path == '.':
		return 0

	def mkdir(p):
		try:
			os.makedirs(p)
			os.chmod(p, mode)
		except OSError as exc:
			if exc.errno == errno.EEXIST:
				pass
			else:
				raise

	if hasattr('__iter__', path):
		[mkdir(p) for p in path]
	else:
		mkdir(path)
	
def merge(arrays):
	''' Takes two recarrays, merges them together and keeps attributes '''
	from numpy import hstack
	return arrays[0].__array_wrap__(hstack(arrays))


def unique( list_of_items, unique=False ):
	''' creates list of unique items '''
	a = sorted( set ( list_of_items ))
	dbg(( list_of_items, unique ), l=9 )
	if not unique:
		return a
	elif unique and len(a) == 1:
		return a[0]
	elif unique and len(a) > 1:
		raise ValueError, 'Too many values present when single exp'
	else:
		dbg(list_of_items)
		raise ValueError, 'Well this shouldn\'t happen'


def subset(d, unique=False, **kwargs):
	'''
	Database selection tool from np.recarray()

	If multiple keywords are pass, it is taken as AND operation
	If list for single option is passed, OR operation (fhr=[0,12,15])
	'''
	options=locals()
	import numpy as np
	for descriptor, desired in kwargs.iteritems(): #options.iteritems():
			#_if desired is an array, convert
			if type(desired) == np.ndarray:
				desired = desired.tolist()

			#_skip special
			if descriptor in ['d',None,'unique','kwargs']: continue

			#_read in values of attributes from data
			values=d.__getattribute__(descriptor)

			#_convert values for comparison
			values=np.array(values)

			#_makes it easier to keep consistent
			indices=np.array((0))
			if type(desired) != list:
				indices=np.where(values==desired)[0]
			else:
				indices=[]
				for d_loop in desired:
					[indices.append(i) for i \
						in np.where(values==d_loop)[0]]
			d = d[indices]

	if unique and d.size==1: return d[0]
	elif unique and d.size==0:
		raise ValueError, 'No values present when single rec expected'
	elif unique and d.size>1:
		print unique, d.size
		raise ValueError, 'Too many values present when one rec exp'
	else: return d


def strictly_increasing(L):
    return all(x<y for x, y in zip(L, L[1:]))


def strictly_decreasing(L):
    return all(x>y for x, y in zip(L, L[1:]))


def non_increasing(L):
    return all(x>=y for x, y in zip(L, L[1:]))


def non_decreasing(L):
    return all(x<=y for x, y in zip(L, L[1:]))


def open_url(url):
	'''remmmiiiinndder'''
	import urllib2
	return urllib2.urlopen(url).read()


def dbg(msg, l=1):
	''' if global debug is set to true, be more verbose '''

	if debug >= l:
		import inspect
		msg = msg.__repr__()
		curf = inspect.currentframe()
		calf = inspect.getouterframes( curf, 2 )
		file, line, method = calf[1][1:4]
		print '[%s.%s.%i] %s' % ( file, method, line, msg )





