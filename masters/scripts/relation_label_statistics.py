from itertools import permutations
import copy

from annotations.bratutils import Relation


def filter_ann_for_relation(annotation, relation_type, entity_pairs):
    total_relations = len(annotation.relations)
    incoming_relations = [r for r in annotation.relations if r.type == relation_type]
    annotation.relations = []
    kept = 0
    for start, end in permutations(annotation.entities, 2):
        if (start.type, end.type) in entity_pairs:
            relation = next((r for r in incoming_relations if r.arg1==start and r.arg2==end), None)
            if relation and relation.type == relation_type:
                annotation.relations.append(copy.deepcopy(relation))
                kept += 1
            else:
                annotation.relations.append(Relation('none', start, end))
    # print(f"Expanded to {len(annotation.relations)} \t\tfrom {total_relations},\t\tkept {kept} relations.")
    return kept, total_relations

def get_all_relation_to_entity(annotations):
    # Builds dictionary
    # Key: relation type
    # Value: set of tuples (Entity1, Entity2)
    relation_entity_set = dict()
    for annotation in annotations:
        for relation in annotation.relations:
            if relation.type not in relation_entity_set:
                relation_entity_set[relation.type] = set()
            relation_entity_set[relation.type].add((relation.arg1.type, relation.arg2.type))
    return relation_entity_set


if __name__ == '__main__':
    from vishnu.paths import path

    from annotations.annmlutils import open_anns
    types = ['MeasuredValue', 'Constraint', 'ParameterSymbol', 'ParameterName',
             'ConfidenceLimit', 'ObjectName', 'Confidence', 'Measurement',
             'Name', 'Property']
    anns = open_anns(path['organn'], types=types, use_labelled=True)

    relation_entity_set = get_all_relation_to_entity(anns)
    relations = list(relation_entity_set.keys())

    for relation in relations:
        anns = open_anns(path['organn'], types=types, use_labelled=True)
        total_kept = 0
        total_sent = 0
        new_labels = 0
        for ann in anns:
            kept, sent = filter_ann_for_relation(ann, relation, relation_entity_set[relation])
            total_kept += kept
            total_sent += sent
            new_labels += len(ann.relations)
        perc = total_kept/new_labels
        print(f"For relation: {relation},\n\t\t\tsent\t\tkept\t\tnlabels\t\tperc"
              f"\n\t\t\t{total_sent}\t\t{total_kept}\t\t{new_labels}\t\t{perc:.2f}")


'''
For relation: Name,
			sent		kept		nlabels		perc
			6026		1158		16345		0.07
For relation: Measurement,
			sent		kept		nlabels		perc
			6026		3500		37915		0.09
For relation: Confidence,
			sent		kept		nlabels		perc
			6026		447		2372		0.19
For relation: Property,
			sent		kept		nlabels		perc
			6026		921		18524		0.05
'''