from datetime import datetime

class StopWatch():

	def __init__(self,memory=False,logfile=None):
		if memory:
			self.ticks = None
		self.logfile = logfile
		self.restart()

	def restart(self):
		self.start = datetime.today()
		self.last = self.start
		if hasattr(self, 'ticks'):
			self.ticks = []

	def tick(self,note=None, report=False):
		firsttick = self.start==self.last
		delta = datetime.today() - self.last
		self.last = datetime.today()
		if hasattr(self, 'ticks'):
			self.ticks.append((delta,note))
		message = 'Tick: ' + str(delta) + (' ('+str(note)+')' if note else '')
		if report:
			print(message, flush=True)
		if self.logfile:
			with open(self.logfile, 'w' if firsttick else 'a') as f:
				f.write(message+'\n')
		return delta

	def total(self):
		return datetime.today() - self.start

	def report(self, prefix=None):

		end = datetime.today()
		#sameday = end.date() == self.start.date()

		if prefix is not None: print(prefix)
		#print('Start:',self.start.time() if sameday else self.start)
		print('Start:', self.start)
		if hasattr(self, 'ticks'):
			for t in self.ticks:
				print('\tTick:',t[0],'('+t[1]+')' if t[1] else '')
		#print('End:  ',end.time() if sameday else end)
		print('End:  ', end)
		print('Total:',self.total(), flush=True)
