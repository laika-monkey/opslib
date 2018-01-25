def newdtg(d, seconds=1, fmt_dtg='%Y-%m-%d-%H-%M-%S', **kwargs):
	''' more date time group forward #seconds '''
	import time

	if type(d) != str:
		raise TypeError, 'NEWDTG input must be of type string'
	
	#_convert to epoch time
	e = dtg2epoch(dtg, fmt_dtg, **kwargs)
	
	e += seconds

	return time.strftime(fmt_dtg, time.gmtime(float(e)))

def dtg2epoch(dtg, fmt_dtg='%Y-%m-%d-%H-%M-%S', **kwargs):
	''' convert from a string time format to epoch seconds '''
	from datetime import datetime
	import calendar

	#_convert single dtg
	if not hasattr(dtg, '__iter__'):
		e =  datetime.strptime(dtg, fmt_dtg)
		e = calendar.timegm((e.year, e.month, e.day,
			e.hour, e.minute, e.second))
		return e 
	
	#_convert array of dtgs
	else:		
		retype = type(dtg)	
		e = [datetime.strptime(d, fmt_dtg) for d in dtg]
		e = [calendar.timegm((f.year, f.month, f.day,
			f.hour, f.minute, f.second)) for f in e]
		return retype(e)

def epoch2dtg(epoch, fmt_dtg='%Y-%m-%d-%H-%M-%S', **kwargs):
	''' convert from epoch time to formatted string '''
	import time

	#_convert single value
	if not hasattr(epoch, '__iter__'):
		return time.strftime(fmt_dtg, time.gmtime(float(epoch)))

	else:
		retype = type(epoch)
		return retype([time.strftime(fmt_dtg, time.gmtime(float(e))) \
				for e in epoch])

def epoch2iso(e):
	''' convert from epoch time to iso format '''
	from datetime import datetime

	if hasattr(e, '__iter__'):
	        return [datetime.utcfromtimestamp(i).isoformat() for i in e]
	else:
	        return datetime.utcfromtimestamp(e).isoformat()

def epoch2local(e):
	from time import localtime, mktime
	from datetime import datetime
     
	#_create initial time tuple
	str_time = localtime(e)

	#_convert to seconds
	time_sec = mktime(str_time)

	#_convert to datetime
	dt = datetime.fromtimestamp(time_sec)

	return dt.isoformat()

def julian2dtg(year, jday, fmt='%Y%m%d%H%M%S', **kwargs):
	''' convert julian to date-time-group '''
	import datetime as dt
	from numpy import array

	if hasattr(jday, '__iter__'):
	        d = []
	        for j in jday:
	                dtg = dt.datetime(int(year), 1, 1)
	                dtg += dt.timedelta(float(j)-1)
	                d.append(dtg.strftime(fmt))
	        d = array(d)
	else:
	        d = dt.datetime(int(year), 1, 1) + dt.timedelta(float(jday)-1)
	        d = d.strftime(fmt)
	return d

def julian2epoch(year, jday, **kwargs):
	''' convert julian to date-time-group '''
	from calendar import timegm
	import datetime as dt
	from numpy import array
	if hasattr(jday, '__iter__'):
	        d = []
	        for j in jday:
	                dtg = dt.datetime(int(year), 1, 1)
	                dtg += dt.timedelta(float(j)-1)
	                d.append(timegm(dtg.timetuple()))
	        d = array(d)
	else:
	        d = dt.datetime(int(year), 1, 1) + dt.timedelta(float(jday)-1)
	        d = timegm(d.timetuple())
	return d

def dtg2iso(dtg, fmt='%Y%m%d%H', **kwargs):
	''' convert YYYYMMDDHH to YYYY-mm-ddTHH:MM:SS '''
	import datetime
	dt = datetime.datetime.strptime(dtg, fmt)
	return dt.isoformat()

def dtg2julian(dtg, fmt='%Y%m%d%H%M%S', **kwargs):
	''' Convert datetimegroup to julian day '''
	import datetime
	dt = datetime.datetime.strptime(dtg, fmt)
	tt = dt.timetuple()
	return tt.tm_yday+(tt.tm_hour*3600 + tt.tm_min*60 + tt.tm_sec) / 86400.




