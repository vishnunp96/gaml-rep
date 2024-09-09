import pandas
import numpy

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score, precision_recall_curve

from itertools import cycle, islice


def plot_surface(ax,df):
	#df = df.fillna(0)

	#print(df)
	#print(df.values)

	Y,X = numpy.meshgrid(df.columns.tolist(),df.index.tolist())

	#print(X)

	#print(Y)

	surf = ax.plot_surface(X, Y, df.values, cmap=None)
	return surf


def make_surface_plot(df,cols=None,xvals=None,yvals=None):
	fig = plt.figure()
	ax = fig.gca(projection='3d')

	plot_surface(ax,make_grid(df,cols=cols,xvals=xvals,yvals=yvals))

	return fig


def make_grid(df,cols=None,xvals=None,yvals=None):
	
	if not cols:
		cols = df.columns
	xcol = cols[0]
	ycol = cols[1]
	zcol = cols[2]

	if not xvals:
		xvals = df[xcol]
	if not yvals:
		yvals = df[ycol]

	xvals = sorted(set(xvals))
	yvals = sorted(set(yvals))

	grid = pandas.DataFrame(index=xvals,columns=yvals)

	foundvalue = True

	for index, row in df.iterrows():

		gotx = not numpy.isnan(row[xcol]) and row[xcol] in grid.index
		goty = not numpy.isnan(row[ycol]) and row[ycol] in grid.columns

		#print(f'Got x: {gotx}, Got y: {goty}')

		if gotx and goty:
			grid.at[row[xcol], row[ycol]] = row[zcol]
		elif goty:
			grid[row[ycol]] = row[zcol]
		elif gotx:
			grid.loc[row[xcol]] = row[zcol]
		else:
			#grid[grid.columns] = row[zcol]
			foundvalue = False

	if not foundvalue and len(df)==1:
		grid[grid.columns] = df.iloc[0][zcol]

	return grid



def plot_roc(y_test, y_score, title='Receiver Operating Characteristic', labels=None, colors=None, filename=None):

	if not isinstance(y_score, list) and not isinstance(y_score,tuple):
		y_score = [y_score]
	if not isinstance(y_test, list) and not isinstance(y_test,tuple):
		y_test = [y_test]
	if len(y_test) != len(y_score):
		raise ValueError('Incorrect numbers of targets/scores provided ('+str(len(y_test))+'/'+str(len(y_score))+').')

	if not labels:
		labels = list(range(1,len(y_score)+1))
	elif not isinstance(labels,list) and not isinstance(labels,tuple):
		if len(y_score)>1:
			raise ValueError('Incorrect number of labels provided.')
		labels = [labels]
	if not colors:
		colors = list(islice(cycle(['darkorange','darkgreen','darkred']), len(y_score)))
	elif not isinstance(colors,list) and not isinstance(colors,tuple):
		colors = [colors]
	if len(colors) != len(y_score):
		colors = list(islice(cycle(colors), len(y_score)))

	plt.figure()
	lw = 2
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

	for y_t,y_s,l,c in zip(y_test,y_score,labels,colors):

		#print(f'y_t: {type(y_t)}, y_s: {type(y_s)}')

		fpr, tpr, _ = roc_curve(y_t, y_s)
		roc_auc = auc(fpr, tpr)

		plt.plot(fpr, tpr, color=c, lw=lw, label= str(l) + ' (area = %0.2f)' % roc_auc)

	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title(title)
	plt.legend(loc="lower right")

	if filename:
		plt.savefig(filename)
		plt.close()
	else:
		plt.show()



def plot_pr(y_test, y_score, filename=None):

	precision, recall, _ = precision_recall_curve(y_test, y_score)
	pr_auc = auc(recall, precision)
	average_precision = average_precision_score(y_test, y_score)

	plt.step(recall, precision, color='b', alpha=0.2,where='post',label='PR curve (area = %0.2f)' % pr_auc)
	plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')

	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.title('Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
	plt.legend(loc="lower left")

	if filename:
		plt.savefig(filename)
		plt.close()
	else:
		plt.show()














