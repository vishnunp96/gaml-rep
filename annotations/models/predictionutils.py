def token_labels_to_entities(predicted_labels, token_idxs):

	ent_type,ent_start,ent_end = None,None,None

	for i,(label,(start,finish)) in enumerate(zip(predicted_labels,token_idxs)):

		if label.endswith('_begin'): # Start new entity
			if ent_type is not None:
				#ann.entity(ent_type,ent_start,ent_end) # Finish previous entity
				yield (ent_type,ent_start,ent_end)
			ent_type = label[:-6] # Remove '_begin' from label
			ent_start,ent_end = start,finish
		elif label.endswith('_inside'):
			if ent_type is None: # No begin label was given, treat as "begin"
				ent_type = label[:-7] # Remove '_inside' from label
				ent_start,ent_end = start,finish
			else: # ent_type is not None
				if ent_type != label[:-7]: # Transitioned to new type
					#ann.entity(ent_type,ent_start,ent_end) # Finish previous entity
					yield (ent_type,ent_start,ent_end)
					ent_type = label[:-7] # Remove '_inside' from label
					ent_start,ent_end = start,finish
				else: # Same type as before
					ent_end = finish
		else: # 'outside' label by default
			if ent_type is not None: # There is an entity which needs finishing
				#ann.entity(ent_type,ent_start,ent_end)
				yield (ent_type,ent_start,ent_end)
				ent_type = None

	#return ann

#todo: need to add token_label_to_entities_map