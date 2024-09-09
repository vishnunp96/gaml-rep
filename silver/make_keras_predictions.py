if __name__ == '__main__':

	from tensorflow.keras.models import load_model
	from tensorflow.keras.preprocessing.sequence import pad_sequences

	from utilities.kerasutils import Projection,MinMaxMeanPool

	import pandas

	from utilities.argparseactions import ArgumentParser,FileAction

	from preprocessing.manifest import ManifestAction
	from annotations.wordembeddings import WordEmbeddings

	from silver.word2vecNNTrainKeras import make_dataset

	parser = ArgumentParser(description='Make predictions from a list of arXIV identifiers in CSV. Results will be appended to DataFrame and saved in specified location.')
	parser.add_argument('data',action=FileAction, mustexist=True,help='CSV file with column of arXiv ids to make predictions for.')
	parser.add_argument('source',action=ManifestAction,help='Source directory containing manifest.json.')
	parser.add_argument('model',action=FileAction, mustexist=True,help='Keras model to use when making predictions.')
	parser.add_argument('embeddings',action=FileAction, mustexist=True,help='Word embeddings file.')
	parser.add_argument('predictions',action=FileAction,mustexist=False,help='Path at which to store model predictions.')
	parser.add_argument('-c','--column',default='id',help='Column from DATA containing arXiv ids. Defaults to \'id\'.')
	args = parser.parse_args()

	data = pandas.read_csv(args.data)

	embeddings = WordEmbeddings.open(args.embeddings)

	dataset = pad_sequences(make_dataset(args.source, data[args.column], -1, embeddings, force=True)[0])

	model = load_model(args.model,custom_objects={i.__name__:i for i in [Projection,MinMaxMeanPool]})

	predictions = model.predict(dataset)

	print(predictions.shape)
	print(data.shape)
	print(dataset.shape)
	print(data[args.column].shape)

	data['prediction'] = predictions

	data.to_csv(args.predictions, index=False)
