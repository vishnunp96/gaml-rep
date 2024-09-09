if __name__ == "__main__":

	from gaml.utilities.argparseactions import ArgumentParser,FileAction
	import pandas
	import matplotlib.pyplot as plt

	parser = ArgumentParser(description="Plot histogram of values in file.")
	parser.add_argument('sourcepath',action=FileAction, mustexist=True, help='Loss values to plot.')
	parser.add_argument('-e','--epochs',type=int, help='Number of epochs to plot.')
	parser.add_argument('-n','--average',type=int,default=10, help='Number of tail rows used to calculate averages.')
	args = parser.parse_args()

	data = pandas.read_csv(args.sourcepath,index_col='epoch',nrows=args.epochs)

	data['train_loss'] = data['pos_train_loss'] + data['neg_train_loss']
	data['test_loss'] = data['pos_test_loss'] + data['neg_test_loss']

	data['train_acc'] = (data['pos_train_acc'] + data['neg_train_acc']) / 2
	data['test_acc'] = (data['pos_test_acc'] + data['neg_test_acc']) / 2

	print(data.tail(args.average).mean())

	#data[['train_loss','test_loss']].plot()
	#data[['pos_train_acc','neg_train_acc','pos_test_acc','neg_test_acc','train_acc','test_acc']].plot()
	#data[['train_acc','test_acc']].plot()

	fig1 = plt.figure()
	loss_ax = fig1.add_subplot('111')
	loss_ax.plot(data['train_loss'],label='Training Set',color='red')
	loss_ax.plot(data['test_loss'],label='Test Set',color='blue')
	loss_ax.legend(loc='best')
	loss_ax.set_xlabel('Epoch')
	loss_ax.set_ylabel('Loss')
	fig1.tight_layout()

	fig2 = plt.figure()
	acc_ax = fig2.add_subplot('111')
	acc_ax.plot(data['pos_train_acc'],label='+ve Training Set',color='green',linestyle='--',linewidth=1)
	acc_ax.plot(data['neg_train_acc'],label='-ve Training Set',color='darkorange',linestyle='--',linewidth=1)
	acc_ax.plot(data['pos_test_acc'],label='+ve Test Set',color='slategrey',linestyle='--',linewidth=1)
	acc_ax.plot(data['neg_test_acc'],label='-ve Test Set',color='saddlebrown',linestyle='--',linewidth=1)
	acc_ax.plot(data['train_acc'],label='Training Set',color='red',linestyle='-',linewidth=2)
	acc_ax.plot(data['test_acc'],label='Testing Set',color='blue',linestyle='-',linewidth=2)
	acc_ax.legend(loc='best')
	acc_ax.set_xlabel('Epoch')
	acc_ax.set_ylabel('Accuracy')
	fig2.tight_layout()

	plt.show()
