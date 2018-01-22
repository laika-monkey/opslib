def newdtg(d, seconds=1, fmt_dtg='%Y-%m-%d-%H-%M-%S', **kwargs):
	''' more date time group forward #seconds '''
	from datetime import datetime
	import calendar
	import time

	if type(d) != str:
		raise TypeError, 'NEWDTG input must be of type string'
	
	#_convert to epoch time
	e = datetime.strptime(d, fmt_dtg)
	e = calendar.timegm((e.year, e.month, e.day, e.hour, e.minute, e.second))
	
	e += seconds

	return time.strftime(fmt_dtg, time.gmtime(float(e)))
