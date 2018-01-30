#_constants
C 	= 2.99792458e8	#_Speed of Light	[m/s]
H	= 6.6260957e-34	#_Planck's Constant	[m2kg/s]
K	= 1.381e-23	#_Boltzmann's Constant	[J/K]
Rd	= 289.058	#_gas constant, dry	[J/kg/K]
Rv	= 461.91	#_gas constant, moist	[J/kg/K]
R	= 8.3144621	#_universal gas costant	[J/kg/mol]
N	= 6.0221413e23	#_Avogadro Constant	[molecules/mole]
Re	= 6.378e6	#_radius of earth (avg)	[m]
G	= 9.81		#_sfc gravitational acc [N/kg]

from numpy import pi as PI

#_standard temperature conversions
def F2C(T): return (T - 32) * 5. / 9.
def C2F(T): return T * 9./5. + 32 #_double, subtract a tenth, add 32
def C2K(T): return T + 273.15
def F2K(T): return C2K(F2C(T))
def F2R(T): raise RuntimeError, 'fuck you'

#_radiation, wavenumber functions are not ANGULAR wavenumbers
def wavenumber2length(k): return 1./k/100	#_K IS IN CM-1
def wavelength2frequency(l): return C/l
def wavenumber2frequency(k): return wavelength2frequency(wavenumber2length(k))

class Molecules(object):
	def __init__(self):
		self.retard = 'strength'
		molecules = {
			'ozone' 	: {
			'molecular_mass' : .0479982, },	#_kg/mol
			'benzene' 	: {
			'molecular_mass' : .07811, }, 
			}		

		for molecule, attributes in molecules.items():
			self.__setattr__(molecule, attributes)
			
############################################################################_80
#_FUNCTIONS_###################################################################
###############################################################################

def planck(T, dom, rayleigh_jeans=False, domain='wavelength'):
	'''
	PASS SI UNITS, RETURNS NON-SI
	dom		float,		wavelength of information {m|m-1|s}
	T		float,		temperature of blackbody
	rayleigh-jeans	bool,		for u-wave, can use R-J approx
	domain		string,		what is the spectral domain
					{wavenumber|length|frequency}
	returns 
	radiance	float|ndarray,	W/sr/m/{um|cm-1|s}
	'''
	from numpy import exp

	#_calculate the emission as a function of wavelength and temp
	if domain == 'wavelength':
		if not rayleigh_jeans:
		    e = exp(H*C/K/dom/T)
		    B = 2*H*C*C/dom**5/(e-1.)
		else:
		    B = 2*C*K*T/dom**4			
		B *= 1e-6 			#_W/m2/sr/um
	elif domain == 'frequency':
		e = exp(H*dom/K/T)
		B = 2*H*dom**3/C/C/(e-1.)	#_W/m2/sr/s
	elif domain == 'wavenumber':
		e = exp(H*C*dom/K/T)
		B = 2*H*C*C*dom**3/(e-1)		
		B *= 100 			#_W/m2/sr/cm
	else:
		raise RuntimeError, 'invalid domain space {}'.format(domain)

	return B
	
def planck_inv(B, dom, domain='wavelength'):
	'''
	PASS IN NON-SI (per um, cm-1)
	B		float(N),	radiances	(W/m2/sr/{um|s|cm-1})
	lamb		float(N),	wavelength	(m)
	rayleigh-jeans	bool,		for u-wave, can use R-J approx
	domain,		string,		'wavelength' or 'frequency'
	
	returns
	Brightness Temperature, float|ndarray,	K
	'''
	from numpy import log

	#_for now
	if B.flatten().size != dom.flatten().size:
		raise RuntimeError, 'sizes must match'
	
	#_inverse planck function
	if domain == 'wavelength':
		T = H*C/dom/K/log(1.+2.*H*C*C/B/1.e6/dom**5.)		
	elif domain == 'frequency':
		T = H*dom/K/log(1.+2.*H*dom**3./C**2./B)
	elif domain == 'wavenumber':
		T = H*C*dom/K/log(1.+2.*H*C*C*dom**3./(B/1.e2))
	else:
		raise RuntimeError, 'invalid domain space {}'.format(domain)

	return T

def beer_lamber(tau, mu=0):
	'''
	Implement Beer-Lambert-Tacokeeno Law
	t=exp(-tau)
	tau	flt,	optical depth of layer
	mu	flt,	zenith angle through layer in
			radians from vertical
	returns
	transmittance	flt
	'''
	from numpy import exp, cos
	return exp(-tau/cos(mu))
	
def size_parameter(r, l):
	'''
	returns size parameter for given particle radii and wavelengths
	r	ndarray(flt),	radii of particle
	l	ndarray(flt),	wavelengths of radiation
	
	x << 1		rayleigh regime
	x ~ 1-100	mie regime, techincally requires spherical particle
	x > 100		geometric ray tracing
	'''
	from numpy import array
	return 2.*PI*(array(r)/array(l))
	
def size_parameter_plot(lrng=(1e-7, 1e-1), rrng=(1e-10, 1e-1), logscale=True):
	'''
	Grant Petty Figure 12.1: Size Parameter
	produces size parameter contour plt
	'''
	from numpy import linspace, meshgrid
	import matplotlib.pyplot as plt
	from matplotlib import cm, ticker
	import matplotlib

	l = linspace(lrng[0], lrng[1], 1e6)	#_wavelengths
	r = linspace(rrng[0], rrng[1], 100)	#_radii

	#_kills memory
	l2, r2 = meshgrid(l, r)		#_make two dimensional arrays
	x = size_parameter(r2, l2)	#_calculate 2d size parameter

	#_initialize plotting area
	ax = plt.figure().add_subplot(111)
	
	#_contour levels
	levels = [0, .002,. 2, 2000] #_Geometric Optics,Mie,Rayleigh,Neg[::-1]
	cn = ax.contourf(l, r, x, locator=ticker.LogLocator(),cmap=cm.jet)
	cx = ax.contour(l, r, x, levels=levels,linestyles='--')
	
	#_plot colorbar and inline labels for noted regimes
	plt.colorbar(cn)
	plt.clabel(cx, use_clabeltext=True)
	
	#_set axes limits and style
	ax.set_xlim(lrng); ax.set_ylim(rrng)
	ax.set_xscale('log'); ax.set_yscale('log')
	
	if logscale:
		loc = [	1e-7, 1e-6,  1e-5,   1e-4, 1e-3, 1e-2,  1e-1]
		lab = ['0.1um', '1um', '10um', '100um', '1mm', '1cm', '10cm']
		ax.set_xticks(loc); ax.set_xticklabels(lab)
	
		loc = [	 1e-9,  1e-8,   1e-7, 1e-6,  1e-5,   1e-4, 1e-3, 1e-2]
		lab = [ '1nm','10nm','0.1um','1um','10um','100um','1mm','1cm']
		ax.set_yticks(loc); ax.set_yticklabels(lab)
	
	ax.set_xlabel('Wavelength'); ax.set_ylabel('Radius')
	ax.set_title('Petty Figure 12.1: Size Parameter')
	plt.savefig('figure_12.1.png')
	
def rh2wv(rh, t, p):
	'''
	convert from relative humidity to vapor mass mixing ratio
	rh	float,	relative humidity, percent
	T	float,	temperature, K
	P	float,	pressure, Pa
	
	returns:
	w	float,	mass mixing ratio in g/kg
	'''
	es = e_s(t)
##	ws = es * Rd / Rv / (p-es)
	ws = .622 * es / p
	return rh * ws / 100 * 1000	#_convert to g/kg
	
def e_s(t):
	'''
	calculate the approximate saturation vapor pressure at T
	T	float,	temperature, K
	'''
	from numpy import exp
	A = 2.53e11	#_Pa
	B = 5420. 	#_K
	return A*exp(-B/t)

def Z2z(Z):
	''' convert geopotential height to altitude '''
	return Re / (Re/Z-1)
##	return Z*9.80/G

def kg2ppmv(m, p, t, solute='ozone', **kwargs):
	'''
	convert mass mixing ratio (kg/kg) to parts per million, volume
	ppmV = 1 V_solute / 1e6 V_solution
	
	m	float,	mass mixing ratio	[kg/kg]
	p	float,	air pressure		[Pa]
	t	float,	air temperature		[K]
	
	returns
	ppmV
	'''
	M	= Molecules()
##	n	= m / M.__getattribute__(solute)['molecular_mass']	#_[kg]
	n	= m / getattr(M, solute)['molecular_mass']		#_[kg]
	vs	= n*R*t/p
	vd	= Rd*t/p
	return vs/vd
'
