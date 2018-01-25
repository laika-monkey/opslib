def eof(d0, d1, centered=True, idx=0, **kwargs):
	'''
	compute regression?  eofs?
	d0, d1	np.matrix,	Datasets, properly aligned d0.shape[0] == d1.shape[0]
	centered,	bool,	Has mean already been removed from data?
	idx,		int,	state space index	
	'''
	import numpy as np
	import libnva as ln
	import  matplotlib.pyplot as plt
	
	#_I haven't removed the variance at each gridpoint so this doesn't make sense 
	#_that is, the var in time is 0
	
	lat, lon = np.linspace(-90,90,181), np.linspace(-180,180,361)
	lat2d, lon2d = np.meshgrid( lat, lon )
	
	#_remove mean if not centered datasets
	if not centered:
		d0 -= d0.mean()
		d1 -= d1.mean()
	
	if idx:
		raise RuntimeError, 'Not implemented yet. Order them yourself for now.'
		#_make array of indices 0-ndim
		#_remove idx from array, put at front
		#_.transpose((order))
	
	#_reshape datasets and convert to matrixes? Matrices?
	shape = d0.shape[1:] #_???
	d0 = np.matrix( d0.reshape( d0.shape[0], np.prod(d0.shape[1:]) ))
	d1 = np.matrix( d1.reshape( d1.shape[0], np.prod(d1.shape[1:]) ))
	n, ndummy = d0.shape
	if d0.shape[0] != d1.shape[0]:
		raise RuntimeError, 'Matrices not alligned '+str((d0.shape, d1.shape))

	covar = d0.T * d1 / (n-1.)
	covar = covar.reshape(shape)
	
	m = ln.draw_map()
	m.contourf( lon2d, lat2d, covar, latlon=True )
	plt.colorbar()
	plt.show()
	
def vector_len(x, **kwargs):
	''' 
	return length of vector 
	np.linalg.norm(x) does the same operation
	'''
	
	import numpy as np
	if type(x) != np.matrixlib.defmatrix.matrix:
		raise RuntimeError, 'only use on matrix'
	x = x.reshape(x.size,1)
	return np.sqrt(x.T*x)
	
def vector_angle(x,y,radian=True,**kwargs):
	''' find angle between to vectors '''
	import numpy as np
	if type(x) != np.matrixlib.defmatrix.matrix:
		raise RuntimeError, 'only use on matrix'
		
	x = x.reshape(x.size,1)
	y = y.reshape(y.size,1)
	
	#_wilkes 9.14
	theta = np.arccos( x.T*y / np.linalg.norm(x) / np.linalg.norm(y) )
	if not radian: theta *= 180/np.pi
	
	return theta[0,0]
	
def vector_shadow(x,y):
	''' lenth of x in direction of y '''
	import numpy as np
	if type(x) != np.matrixlib.defmatrix.matrix:
		raise RuntimeError, 'only use on matrix'
		
	x = x.reshape(x.size,1)
	y = y.reshape(y.size,1)
	
	#_wilkes 9.16
	L = x.T*y / np.linalg.norm(y)
	
	return L[0,0]
	
def covar_matrix(x,y,**kwargs):
	'''
	np.cov() does this, do not use.
	Wilkes 9.30
	
	Computer the covariance matrix of datasets
	x, y	np.matrix,	data to covar, must have same sample space dimension
	centered,	bool,	was mean removed across sample space
	sample,		int,	dimension of sample space 
	
	Sample space is currently assume to be the first axis
	'''
	x0, x1 = x.shape
	y0, y1 = y.shape
	
	if x0 != y0:
		raise RuntimeError, 'sample space not aligned'+str((x.shape,y.shape))
		
	x = center_data(x, **kwargs)
	y = center_data(y, **kwargs)
		
	#_calculate covariance matrix
	return x.T*y / (x0-1.)
	
def correlation_matrix(x,**kwargs):
	'''
	calc correlation matrix 
	x	np.array/matrix,	axis=0, sample space; axis=1, state space
	
	not sure what this actually is
	
	Wilkes 9.31
	'''
	import numpy as np
	
##	x = center_data(x, **kwargs)
	
	#_get covariance matrix and a diagonal of only the state variances
	s = np.sqrt( np.cov(x) )
	d = np.matrix( np.identity( np.sqrt(s.size) ) * s )

	#_calculate corr matrix
	return d.I*s*d.I
	
def center_data(x, centered=True, sample_index=0, **kwargs):
	''' remove mean from data if not centered, assumes sample spaceaxis=0 '''
	if not centered: x -= x.mean(axis=sample_index)
	return x
	
def remove_climo( data, dt=12 ):
	''' 
	returns data with dt signal removed
	
	data	np.ndarray,	Contains data with time dimension in 0 axis
	dt		int,		number to skip in cycle (e.g., 12 for monthly)
	'''
	import numpy as np
	
	#_take mean along first axis at dt rate
	climo = np.array([ data[i::dt].mean(axis=0) for i in xrange(12) ])
	
	#_build array in which to repeat climatology for removal
	shape = tuple( [data.shape[0]/dt]+[1] * (data.ndim-1) )

	#_remove signal form data
	data -= np.tile( climo, shape )

	return data, climo
	
def detrend( data, fit='linear', order=1, breakpoint=False ):
	''' 
	remove trends from data vector.  If data.ndim > 1, detrend columns
	
	fit,	string,		'linear' or 'constant'   Linear removes a best fit
						line while constant removes the mean
	breakpoint,	int,	Not implemented.  Allow for trending regions to be
						removed instead of over whole set
	order		int,	power of fit
	
	'''
	import numpy as np
	
	if fit == 'linear':
		size = data.shape			#_store to return data to size
		
		#_if only one vector, add dummy dimension
		if data.ndim == 1	: column = np.array( data.reshape( data.size, 1 ))
		else				: column = np.array( data.copy() )
		
		#_loop over each column and remove trend
		for n in range( column.shape[1] ):
			y_o = column[:,n]			#_pull y values
			x_o = range( y_o.size )		#_generate x values

			fit = np.polyfit( x_o, y_o, order )	#_get first order fit params
			f_x = np.poly1d( fit )				#_use them to build a function
			column[:,n] = y_o - f_x(x_o)		#_remove from data

		column = column.reshape(size)	#_return to original shape
		
	elif fit == 'constant':
		pass
	else:
		raise RuntimeError, error, 'Undefined type of trend'
	
	return column

	
def autocorrelation(x, y=None, lag=10, **kwargs):
	'''
	x	np.ndarray	autocorrelate at lag n
	y	np.ndarray	crosscorrelate at lag n
	lag	int			interval between corr
	
	autocorrelations return one tail
	'''
	from numpy import array, corrcoef, arange
	from scipy.stats import pearsonr
	
	if y == None:
		return array( [1] + [pearsonr(x[:-i], x[i:])[0] 
			for i in arange(1,lag+1) ] )
			
	else:	#_perform cross correlation
		tmp = []
		for n in arange(lag*2+1)-lag:
			if n<0		: 
				tmp.append( pearsonr(x[-n:],y[:n])[0] )
			elif n==0	: 
				tmp.append( pearsonr(x,y)[0] )
			elif n>0	: 
				tmp.append( pearsonr(x[:-n],y[n:])[0] )
			else		: raise RuntimeError, 'What?'
		return array(tmp)		


def dummy_sbatch(cmd,
	job_name='dummy',
	sname='sbatch_{0}_{1}',
	unique_id='id-fill',
	time_sbatch='2:00:00',
	nice=False,
	**kwargs):
	''' write a basic sbatch to get something going '''
	from os import getpid, environ, path
	from time import gmtime, strftime

	time = strftime('%Y-%m-%dT%H-%M-%S', gmtime())
	pid = getpid()
	unique_id = '{0}_{1}_p{2}'.format(unique_id, time, pid)
	sname = sname.format(job_name, unique_id)
	
	out =  '#!/bin/bash\n'
	out += '#SBATCH --job-name={0}_{1}\n'.format(job_name, unique_id)
	out += '#SBATCH --partition=all\n'
	out += '#SBATCH --share\n'
	out += '#SBATCH --time={0}\n'.format(time_sbatch)
	out += '#SBATCH --ntasks=1\n'
	out += '#SBATCH --cpus-per-task=1\n'
	out += '#SBATCH --output=/odyssey/scratch/%u/logs/dummy_%A.txt\n'
	if nice:
		out += '#SBATCH --nice={0:d}\n'.format(nice)
	out += 'module purge\n'
	out += 'module load license_intel\n'
	out += 'module load impi\n'
	out += 'module load intel/15.0-2\n'
	out += 'module load hdf5/1.8.14\n'
	out += 'module load netcdf4/4.3.3\n'
	out += 'module load anaconda27/base_2015a_sharppy13\n'
##	out += 'export TMPDIR=${{SLURM_JOB_NAME}}.${{SLURM_JOB_ID}}\n'
##	out += 'mkdir -p $TMPDIR\n'
	out += '\n'.join(['export {0}={1}'.format(var, environ[var]) \
			for var in ['PYTHONPATH', 'PRODUCTS', 'WORK', 'PATH', 'LOG']])
	out += '\n'
	out += 'source activate my_root\n'
	out += 'echo `which python`\n'
	out += '{0}\n'.format(cmd)

	#_get unique identifiers
	sname = path.join(environ['WORK'], 'batch_scripts', sname)
	with open(sname, 'w') as f:
		f.write(out)

	print sname
	return sname
