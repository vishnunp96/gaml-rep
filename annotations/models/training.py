from utilities.torchutils import train # predict_from_dataloader

import matplotlib.pyplot as plt

import numpy
import pandas
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support,f1_score,precision_score,recall_score

from annotations.bratutils import Standoff
from annotations.metrics import ann_entity_overlap_score

from utilities.mlutils import split_data
from utilities.torchutils import unpack_sequence, cat_packed_sequences, predict_from_dataloader
import os
import itertools

import torch.nn.functional as F

def perform_training(make_model, data, modeldir, make_metrics, make_opt, make_loss_func, batch_size, epochs, patience=None, min_delta=0.0, modelname=None, train_fractions=False, stopwatch=None):

	ann_train,ann_dev,ann_test = data

	if modelname is None:
		modelname = os.path.basename(modeldir)

	if train_fractions:
		plt.figure()
		legend = []
		for i in [0.25,0.5,0.75,1.0]:
			model = make_model()

			train_subset = split_data(ann_train, i, random_state=42)

			train_dataset = model.make_dataset(train_subset)
			dev_dataset = model.make_dataset(ann_dev)
			test_dataset = model.make_dataset(ann_test)

			if stopwatch: stopwatch.tick(f'Created model and constructed datasets for fraction {i:.2f}: {len(train_dataset)} training samples, {len(dev_dataset)} dev samples, {len(test_dataset)} testing samples',report=True)

			# Setup training
			dataloader = train_dataset.dataloader(batch_size=batch_size,shuffle=True)
			dev_dataloader = dev_dataset.dataloader(batch_size=batch_size)

			opt = make_opt(model)
			loss_func = make_loss_func(model, train_dataset)

			if stopwatch: stopwatch.tick(f'Setup training for fraction {i:.2f}',report=True)

			history = train(model, dataloader, epochs, opt, loss_func, dev_dataloader, metrics=make_metrics(model), verbose=1)

			if stopwatch: stopwatch.tick(f'Completed training for fraction {i:.2f}',report=True)

			plt.plot(history['loss'])
			legend.append(f'Train Loss ({i:.2f}, {len(train_subset)})')
			plt.plot(history['val_loss'])
			legend.append(f'Test Loss ({i:.2f}, {len(train_subset)})')

		plt.title('Model Loss With Training Set Fraction')
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		plt.legend(legend, loc='upper right')
		plt.savefig(os.path.join(modeldir,'subset_loss_curves.png'))
	else:
		# Create model
		model = make_model()
		print(model)

		train_dataset = model.make_dataset(ann_train)
		dev_dataset = model.make_dataset(ann_dev)
		test_dataset = model.make_dataset(ann_test)

		if stopwatch: stopwatch.tick(f'Created model and constructed datasets: {len(train_dataset)} training samples, {len(dev_dataset)} dev samples, {len(test_dataset)} testing samples',report=True)

		opt = make_opt(model)
		loss_func = make_loss_func(model, train_dataset)

		# Setup training
		dataloader = train_dataset.dataloader(batch_size=batch_size,shuffle=True)
		dev_dataloader = dev_dataset.dataloader(batch_size=batch_size)

		if stopwatch: stopwatch.tick('Setup training',report=True)

		history = train(model, dataloader, epochs, opt, loss_func, dev_dataloader, metrics=make_metrics(model), patience=25, min_delta=0.001, verbose=1)

		model.save(os.path.join(modeldir,modelname+'.pt'))
		pandas.DataFrame(history).set_index('epoch').to_csv(os.path.join(modeldir,'logs.csv'))

	if stopwatch: stopwatch.tick(f'Completed training ({len(history["epoch"])-1} epochs)',report=True)

	return model, opt, history

def mla_cnn_training(make_model, data, modeldir, make_metrics, make_opt, make_loss_func, batch_size, epochs, relation_embedding, patience=None, min_delta=0.0, modelname=None, train_fractions=False, stopwatch=None):

	ann_train,ann_dev,ann_test = data

	if modelname is None:
		modelname = os.path.basename(modeldir)

	if train_fractions:
		plt.figure()
		legend = []
		for i in [0.25,0.5,0.75,1.0]:
			model = make_model()

			train_subset = split_data(ann_train, i, random_state=42)

			train_dataset = model.make_dataset(train_subset)
			dev_dataset = model.make_dataset(ann_dev)
			test_dataset = model.make_dataset(ann_test)

			if stopwatch: stopwatch.tick(f'Created model and constructed datasets for fraction {i:.2f}: {len(train_dataset)} training samples, {len(dev_dataset)} dev samples, {len(test_dataset)} testing samples',report=True)

			# Setup training
			dataloader = train_dataset.dataloader(batch_size=batch_size,shuffle=True)
			dev_dataloader = dev_dataset.dataloader(batch_size=batch_size)

			opt = make_opt(model)
			loss_func = make_loss_func(model, train_dataset)

			if stopwatch: stopwatch.tick(f'Setup training for fraction {i:.2f}',report=True)

			history = train(model, dataloader, epochs, opt, loss_func, dev_dataloader, metrics=make_metrics(model), verbose=1)

			if stopwatch: stopwatch.tick(f'Completed training for fraction {i:.2f}',report=True)

			plt.plot(history['loss'])
			legend.append(f'Train Loss ({i:.2f}, {len(train_subset)})')
			plt.plot(history['val_loss'])
			legend.append(f'Test Loss ({i:.2f}, {len(train_subset)})')

		plt.title('Model Loss With Training Set Fraction')
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		plt.legend(legend, loc='upper right')
		plt.savefig(os.path.join(modeldir,'subset_loss_curves.png'))
	else:
		# Create model
		model = make_model()
		print(model)

		train_dataset = model.make_dataset(ann_train)
		dev_dataset = model.make_dataset(ann_dev)
		test_dataset = model.make_dataset(ann_test)

		if stopwatch: stopwatch.tick(f'Created model and constructed datasets: {len(train_dataset)} training samples, {len(dev_dataset)} dev samples, {len(test_dataset)} testing samples',report=True)

		opt = make_opt(model)
		loss_func = make_loss_func(relation_embedding)

		# Setup training
		dataloader = train_dataset.dataloader(batch_size=batch_size,shuffle=True)
		dev_dataloader = dev_dataset.dataloader(batch_size=batch_size)

		if stopwatch: stopwatch.tick('Setup training',report=True)

		history = train(model, dataloader, epochs, opt, loss_func, dev_dataloader, metrics=make_metrics(model), patience=25, min_delta=0.001, verbose=1)

		model.save(os.path.join(modeldir,modelname+'.pt'))
		pandas.DataFrame(history).set_index('epoch').to_csv(os.path.join(modeldir,'logs.csv'))

	if stopwatch: stopwatch.tick(f'Completed training ({len(history["epoch"])-1} epochs)',report=True)

	return model, opt, history


def evaluate_entities(model, ann_test, batch_size=1):
	with model.evaluation():

		dataset = model.make_dataset(ann_test)

		test_subjects = dataset.anns[0:2]
		subjects_dataset = model.make_dataset(test_subjects)
		predict_subjects = predict_from_dataloader(model, subjects_dataset.dataloader(batch_size=batch_size))
		subject_preds = [F.softmax(i,dim=1).cpu().detach().numpy() for i in unpack_sequence(predict_subjects)]

		print_len = max(len(c) for c in model.labels.classes_) + 2
		for subject,preds in zip(test_subjects,subject_preds):
			print(f'Lengths: {subject.labels.shape}, {preds.shape}')
			sub_labels = model.labels.inverse_transform(preds.argmax(1))
			for (token,label),pred,pred_lab in zip(subject,preds,sub_labels):
				vec_str = ', '.join(f'{p:.2f}' for p in pred)
				print(('{0:20} {1:'+str(print_len)+'} {2:'+str(print_len)+'} [{3}]').format(token,label,pred_lab,vec_str))
		print(f'[{", ".join(str(i) for i in model.labels.classes_)}]')

		predict_test = predict_from_dataloader(model, dataset.dataloader(batch_size=batch_size))

		test_preds = numpy.vstack([F.softmax(i,dim=1).cpu().detach().numpy() for i in unpack_sequence(predict_test)])
		print(test_preds.shape)
		predicted_test_labels = model.labels.inverse_transform(test_preds.argmax(1))

		test_y = cat_packed_sequences([b_y for b_x,b_y in dataset.dataloader(batch_size=batch_size)])

		val_target = numpy.hstack([i.cpu().detach().numpy() for i in unpack_sequence(test_y)])
		print(val_target.shape)
		true_test_labels = model.labels.inverse_transform(val_target)

		confusion = confusion_matrix(true_test_labels, predicted_test_labels, labels=model.labels.classes_)
		print(confusion)
		print(confusion.shape)
		p,r,f,s = precision_recall_fscore_support(true_test_labels, predicted_test_labels, labels=model.labels.classes_, zero_division=0)

		print(('{0:'+str(print_len)+'} {1:6} {2:6} {3:6} {4:6}').format('TYPE','PREC','REC','F1','Count'))
		for i,label in enumerate(model.labels.classes_):
			print(('{0:'+str(print_len)+'} {1:<6.2f} {2:<6.2f} {3:<6.2f} {4:6d}').format(label,p[i],r[i],f[i],s[i]))

		f1micro = f1_score(true_test_labels,predicted_test_labels,labels=model.labels.classes_, average='micro', zero_division=0)
		f1weighted = f1_score(true_test_labels,predicted_test_labels,labels=model.labels.classes_, average='weighted', zero_division=0)
		f1macro = f1_score(true_test_labels,predicted_test_labels,labels=model.labels.classes_, average='macro', zero_division=0)
		print(f'F1 score: {f1micro:.2f} (micro), {f1macro:.2f} (macro), {f1weighted:.2f} (weighted)')

		precision = precision_score(true_test_labels,predicted_test_labels,labels=model.labels.classes_, average='micro', zero_division=0)
		recall = recall_score(true_test_labels,predicted_test_labels,labels=model.labels.classes_, average='micro', zero_division=0)
		print(f'Precision score: {precision:.2f}, Recall score: {recall:.2f}')

		empty_anns = [('','',Standoff(a.text,[],[],[],[])) for a in ann_test]
		standoff_ann_test = [l.standoff for l in ann_test]
		predicted_ann_test = [a for i,j,a in model.predict(empty_anns,batch_size=batch_size,inplace=True)]
		overlap_score_lenient = ann_entity_overlap_score(predicted_ann_test,standoff_ann_test,strict=False)
		overlap_score_strict = ann_entity_overlap_score(predicted_ann_test,standoff_ann_test,strict=True)
		print(f'Entity overlap score: {overlap_score_strict:.2f} (strict), {overlap_score_lenient:.2f} (lenient)')

		overall_metrics = pandas.DataFrame([['f1micro',f1micro],['f1macro',f1macro],['f1weighted',f1weighted],['precision',precision],['recall',recall],['overlap_strict',overlap_score_strict],['overlap_lenient',overlap_score_lenient]],columns=['score','value'])
		class_metrics = pandas.DataFrame([p,r,f,s],columns=model.labels.classes_,index=pandas.Index(['Precision','Recall','F1','Support'],name='Label')).T

		return class_metrics,overall_metrics



def evaluate_entities_bert(model, ann_test, batch_size=1):
	with model.evaluation():

		dataset = model.make_dataset(ann_test)

		print_len = max(len(c) for c in model.labels.classes_) + 2

		predict_test = predict_from_dataloader(model, dataset.dataloader(batch_size=batch_size))
		sm_pred_lists = [F.softmax(i,dim=1).cpu().detach().numpy() for i in unpack_sequence(predict_test)]
		pred_label_lists = [model.labels.inverse_transform(i.argmax(1)) for i in sm_pred_lists]
		predicted_test_labels = []
		for i, pr_label in enumerate(pred_label_lists):
			im = dataset.index_maps[i]
			predicted_test_labels.extend(
				[pr_label[im[j]] for j in range(max(im.keys()) + 1)])

		print(len(predicted_test_labels))

		test_y = cat_packed_sequences([b_y for b_x,b_y in dataset.dataloader(batch_size=batch_size)])
		target_lists = [i.cpu().detach().numpy() for i in unpack_sequence(test_y)]
		target_labelled_lists = [model.labels.inverse_transform(i) for i in target_lists]
		true_test_labels = []
		for i, t_list in enumerate(target_labelled_lists):
			im = dataset.index_maps[i]
			true_test_labels.extend(
				[t_list[im[j]] for j in range(max(im.keys()) + 1)])

		print(len(true_test_labels))

		confusion = confusion_matrix(true_test_labels, predicted_test_labels, labels=model.labels.classes_)
		print(confusion)
		print(confusion.shape)
		p,r,f,s = precision_recall_fscore_support(true_test_labels, predicted_test_labels, labels=model.labels.classes_, zero_division=0)

		print(('{0:'+str(print_len)+'} {1:6} {2:6} {3:6} {4:6}').format('TYPE','PREC','REC','F1','Count'))
		for i,label in enumerate(model.labels.classes_):
			print(('{0:'+str(print_len)+'} {1:<6.2f} {2:<6.2f} {3:<6.2f} {4:6d}').format(label,p[i],r[i],f[i],s[i]))

		f1micro = f1_score(true_test_labels,predicted_test_labels,labels=model.labels.classes_, average='micro', zero_division=0)
		f1weighted = f1_score(true_test_labels,predicted_test_labels,labels=model.labels.classes_, average='weighted', zero_division=0)
		f1macro = f1_score(true_test_labels,predicted_test_labels,labels=model.labels.classes_, average='macro', zero_division=0)
		print(f'F1 score: {f1micro:.2f} (micro), {f1macro:.2f} (macro), {f1weighted:.2f} (weighted)')

		precision = precision_score(true_test_labels,predicted_test_labels,labels=model.labels.classes_, average='micro', zero_division=0)
		recall = recall_score(true_test_labels,predicted_test_labels,labels=model.labels.classes_, average='micro', zero_division=0)
		print(f'Precision score: {precision:.2f}, Recall score: {recall:.2f}')

		empty_anns = [('','',Standoff(a.text,[],[],[],[])) for a in ann_test]
		standoff_ann_test = [l.standoff for l in ann_test]
		predicted_ann_test = [a for i,j,a in model.predict(empty_anns,batch_size=batch_size,inplace=True)]
		overlap_score_lenient = ann_entity_overlap_score(predicted_ann_test,standoff_ann_test,strict=False)
		overlap_score_strict = ann_entity_overlap_score(predicted_ann_test,standoff_ann_test,strict=True)
		print(f'Entity overlap score: {overlap_score_strict:.2f} (strict), {overlap_score_lenient:.2f} (lenient)')

		overall_metrics = pandas.DataFrame([['f1micro',f1micro],['f1macro',f1macro],['f1weighted',f1weighted],['precision',precision],['recall',recall],['overlap_strict',overlap_score_strict],['overlap_lenient',overlap_score_lenient]],columns=['score','value'])
		class_metrics = pandas.DataFrame([p,r,f,s],columns=model.labels.classes_,index=pandas.Index(['Precision','Recall','F1','Support'],name='Label')).T

		return class_metrics,overall_metrics


def evaluate_relations_old(model, ann_test, batch_size=1):
	with model.evaluation():

		dataset = model.make_dataset(ann_test)

		dataloader = dataset.dataloader(batch_size=batch_size)
		predicted = predict_from_dataloader(model, dataloader)
		predicted_labels = model.labels.inverse_transform(predicted.argmax(1))

		confusion = confusion_matrix(dataset.y_labels, predicted_labels, labels=model.labels.classes_)
		print(confusion)
		print(confusion.shape)
		p,r,f,s = precision_recall_fscore_support(dataset.y_labels, predicted_labels, labels=model.labels.classes_)

		print_len = max(len(c) for c in model.labels.classes_) + 2
		print(('{0:'+str(print_len)+'} {1:6} {2:6} {3:6} {4:6}').format('TYPE','PREC','REC','F1','Count'))
		for i,label in enumerate(model.labels.classes_):
			print(('{0:'+str(print_len)+'} {1:<6.2f} {2:<6.2f} {3:<6.2f} {4:6d}').format(label,p[i],r[i],f[i],s[i]))

		f1micro = f1_score(dataset.y_labels,predicted_labels,labels=model.labels.classes_, average='micro', zero_division=0)
		f1weighted = f1_score(dataset.y_labels,predicted_labels,labels=model.labels.classes_, average='weighted', zero_division=0)
		f1macro = f1_score(dataset.y_labels,predicted_labels,labels=model.labels.classes_, average='macro', zero_division=0)
		print(f'F1 score: {f1micro:.2f} (micro), {f1macro:.2f} (macro), {f1weighted:.2f} (weighted)')

		return pandas.DataFrame([p,r,f,s],columns=model.labels.classes_,index=pandas.Index(['Precision','Recall','F1','Support'],name='Label')).T

def evaluate_relations(model, ann_test, allowed_relations, batch_size=1):
	## This method should be independent of any internal mechanisms of the model,
	## as it relies directly on the Standoff outputs.
	## (It is actually more accurate than the previous version, as it accounts
	## for directionality of relations)
	with model.evaluation():

		anns = [a.standoff for a in ann_test]
		entity_only_anns = [('','',Standoff(a.text,a.entities,[],[],[])) for a in anns]

		evaluation_types = model.labels.classes_

		ground_truth = []
		predictions = []
		for original_ann,(p,i,ann) in zip(anns,model.predict(entity_only_anns, batch_size=batch_size,inplace=True)):
			for start,end in itertools.permutations(original_ann.entities,2):
				if (start.type,end.type) in allowed_relations:
					original_rel = original_ann.get_relation(start,end)
					ground_truth.append(original_rel.type if original_rel and original_rel.type in evaluation_types else 'none')
					predicted_rel = ann.get_relation(start,end)
					predictions.append(predicted_rel.type if predicted_rel else 'none')

		confusion = confusion_matrix(ground_truth, predictions, labels=evaluation_types)
		print(confusion)
		print(confusion.shape)
		p,r,f,s = precision_recall_fscore_support(ground_truth, predictions, labels=evaluation_types, zero_division=0)

		print_len = max(len(c) for c in evaluation_types) + 2
		print(('{0:'+str(print_len)+'} {1:6} {2:6} {3:6} {4:6}').format('TYPE','PREC','REC','F1','Count'))
		for i,label in enumerate(evaluation_types):
			print(('{0:'+str(print_len)+'} {1:<6.2f} {2:<6.2f} {3:<6.2f} {4:6d}').format(label,p[i],r[i],f[i],s[i]))

		f1micro = f1_score(ground_truth,predictions,labels=evaluation_types, average='micro', zero_division=0)
		f1weighted = f1_score(ground_truth,predictions,labels=evaluation_types, average='weighted', zero_division=0)
		f1macro = f1_score(ground_truth,predictions,labels=evaluation_types, average='macro', zero_division=0)
		print(f'F1 score: {f1micro:.2f} (micro), {f1macro:.2f} (macro), {f1weighted:.2f} (weighted)')

		precision = precision_score(ground_truth,predictions,labels=evaluation_types, average='micro', zero_division=0)
		recall = recall_score(ground_truth,predictions,labels=evaluation_types, average='micro', zero_division=0)
		print(f'Precision score: {precision:.2f}, Recall score: {recall:.2f}')

		overall_metrics = pandas.DataFrame([['f1micro',f1micro],['f1macro',f1macro],['f1weighted',f1weighted],['precision',precision],['recall',recall]],columns=['score','value'])
		class_metrics = pandas.DataFrame([p,r,f,s],columns=model.labels.classes_,index=pandas.Index(['Precision','Recall','F1','Support'],name='Label')).T

		return class_metrics,overall_metrics

def evaluate_attributes(model, ann_test, batch_size=1):
	with model.evaluation():

		anns = [a.standoff for a in ann_test]
		stripped_anns = [('','',Standoff(a.text,a.entities,a.relations,[],[])) for a in anns]

		evaluation_types = model.labels.classes_

		ground_truth = []
		predictions = []
		for original_ann,(p,i,ann) in zip(anns,model.predict(stripped_anns, batch_size=batch_size, inplace=True)):
			for ent in original_ann.entities:
				original_attrs = [att for att in original_ann.get_attributes(ent) if att.type in evaluation_types]
				if original_attrs:
					ground_truth.append(original_attrs[0].type) ## TODO: What to do in case of multiple attributes?
				else:
					ground_truth.append('none')
				predicted_attrs = ann.get_attributes(ent) ## TODO: What to do in case of multiple attributes?
				predictions.append(predicted_attrs[0].type if predicted_attrs else 'none')

		confusion = confusion_matrix(ground_truth, predictions, labels=evaluation_types)
		print(confusion)
		print(confusion.shape)
		p,r,f,s = precision_recall_fscore_support(ground_truth, predictions, labels=evaluation_types, zero_division=0)

		print_len = max(len(c) for c in evaluation_types) + 2
		print(('{0:'+str(print_len)+'} {1:6} {2:6} {3:6} {4:6}').format('TYPE','PREC','REC','F1','Count'))
		for i,label in enumerate(evaluation_types):
			print(('{0:'+str(print_len)+'} {1:<6.2f} {2:<6.2f} {3:<6.2f} {4:6d}').format(label,p[i],r[i],f[i],s[i]))

		f1micro = f1_score(ground_truth,predictions,labels=evaluation_types, average='micro', zero_division=0)
		f1weighted = f1_score(ground_truth,predictions,labels=evaluation_types, average='weighted', zero_division=0)
		f1macro = f1_score(ground_truth,predictions,labels=evaluation_types, average='macro', zero_division=0)
		print(f'F1 score: {f1micro:.2f} (micro), {f1macro:.2f} (macro), {f1weighted:.2f} (weighted)')

		precision = precision_score(ground_truth,predictions,labels=evaluation_types, average='micro', zero_division=0)
		recall = recall_score(ground_truth,predictions,labels=evaluation_types, average='micro', zero_division=0)
		print(f'Precision score: {precision:.2f}, Recall score: {recall:.2f}')

		overall_metrics = pandas.DataFrame([['f1micro',f1micro],['f1macro',f1macro],['f1weighted',f1weighted],['precision',precision],['recall',recall]],columns=['score','value'])
		class_metrics = pandas.DataFrame([p,r,f,s],columns=model.labels.classes_,index=pandas.Index(['Precision','Recall','F1','Support'],name='Label')).T

		return class_metrics,overall_metrics



def evaluate_cnn_relations(model, ann_test, relation_labels, batch_size=1):
	with model.evaluation():

		predicted_vectors, predicted_classes, actual_vectors, actual_classes = model.predict(ann_test, batch_size=batch_size)
		ground_truth = actual_classes
		predictions = predicted_classes

		confusion = confusion_matrix(ground_truth, predictions, labels=relation_labels)
		print(confusion)
		print(confusion.shape)
		p,r,f,s = precision_recall_fscore_support(ground_truth, predictions, labels=relation_labels, zero_division=0)

		print_len = max(len(c) for c in relation_labels) + 2
		print(('{0:'+str(print_len)+'} {1:6} {2:6} {3:6} {4:6}').format('TYPE','PREC','REC','F1','Count'))
		for i,label in enumerate(relation_labels):
			print(('{0:'+str(print_len)+'} {1:<6.2f} {2:<6.2f} {3:<6.2f} {4:6d}').format(label,p[i],r[i],f[i],s[i]))

		f1micro = f1_score(ground_truth,predictions,labels=relation_labels, average='micro', zero_division=0)
		f1weighted = f1_score(ground_truth,predictions,labels=relation_labels, average='weighted', zero_division=0)
		f1macro = f1_score(ground_truth,predictions,labels=relation_labels, average='macro', zero_division=0)
		print(f'F1 score: {f1micro:.2f} (micro), {f1macro:.2f} (macro), {f1weighted:.2f} (weighted)')

		precision = precision_score(ground_truth,predictions,labels=relation_labels, average='micro', zero_division=0)
		recall = recall_score(ground_truth,predictions,labels=relation_labels, average='micro', zero_division=0)
		print(f'Precision score: {precision:.2f}, Recall score: {recall:.2f}')

		overall_metrics = pandas.DataFrame([['f1micro',f1micro],['f1macro',f1macro],['f1weighted',f1weighted],['precision',precision],['recall',recall]],columns=['score','value'])
		class_metrics = pandas.DataFrame([p,r,f,s],columns=relation_labels,index=pandas.Index(['Precision','Recall','F1','Support'],name='Label')).T

		return class_metrics,overall_metrics


def evaluate_relations_simple(model, ann_test, batch_size=1):
	with model.evaluation():
		predictions, ground_truth = model.predict(ann_test, batch_size=batch_size)
		evaluation_types = model.labels.classes_
		confusion = confusion_matrix(ground_truth, predictions, labels=evaluation_types)
		print(confusion)
		print(confusion.shape)
		p,r,f,s = precision_recall_fscore_support(ground_truth, predictions, labels=evaluation_types, zero_division=0)

		print_len = max(len(c) for c in evaluation_types) + 2
		print(('{0:'+str(print_len)+'} {1:6} {2:6} {3:6} {4:6}').format('TYPE','PREC','REC','F1','Count'))
		for i,label in enumerate(evaluation_types):
			print(('{0:'+str(print_len)+'} {1:<6.2f} {2:<6.2f} {3:<6.2f} {4:6d}').format(label,p[i],r[i],f[i],s[i]))

		f1micro = f1_score(ground_truth,predictions,labels=evaluation_types, average='micro', zero_division=0)
		f1weighted = f1_score(ground_truth,predictions,labels=evaluation_types, average='weighted', zero_division=0)
		f1macro = f1_score(ground_truth,predictions,labels=evaluation_types, average='macro', zero_division=0)
		print(f'F1 score: {f1micro:.2f} (micro), {f1macro:.2f} (macro), {f1weighted:.2f} (weighted)')

		precision = precision_score(ground_truth,predictions,labels=evaluation_types, average='micro', zero_division=0)
		recall = recall_score(ground_truth,predictions,labels=evaluation_types, average='micro', zero_division=0)
		print(f'Precision score: {precision:.2f}, Recall score: {recall:.2f}')

		overall_metrics = pandas.DataFrame([['f1micro',f1micro],['f1macro',f1macro],['f1weighted',f1weighted],['precision',precision],['recall',recall]],columns=['score','value'])
		class_metrics = pandas.DataFrame([p,r,f,s],columns=model.labels.classes_,index=pandas.Index(['Precision','Recall','F1','Support'],name='Label')).T

		return class_metrics,overall_metrics