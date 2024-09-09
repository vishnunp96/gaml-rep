import numpy

from annotations.brattowindow import StandoffLabels
from annotations.datasets import AnnotationDataset
import itertools
import torch
import torch.nn.utils.rnn as rnn

'''
You get:
x
    - tokenized input ids
    - tokenized attention mask
    - one-hot encoded entity labels for each token in sentence
    - one-hot encoded entity encodings for entities present in sentence
    - reverse flag for relation direction
'''


# Modelled after RelationSplitSpansDirectedLabelledIndexesDataset
class SpecificRelationBertDataset(AnnotationDataset):
    def __init__(self, ann_list, tokenizer, relation_encoder,
                 entity_encoder, relation_focus, allowed_entity_pairs,
                 window_pad, cuda=False,
                 outside_class="outside",
                 begin_class="begin",
                 inside_class="inside"
                 ):
        super(SpecificRelationBertDataset, self).__init__(ann_list, cuda)

        self.relation_focus = relation_focus
        self.allowed_entity_pairs = allowed_entity_pairs
        xs = []
        ys = []

        self.class_template = {"O": outside_class, "B": begin_class,
                               "I": inside_class}
        self.tokenizer = tokenizer

        for a in self.anns:
            tokens, entity_labels, idx_mapping = self.tokenize_and_align(a.tokens, a.labels)
            stripped_labels = [l.split('_')[0] for l in entity_labels]
            onehot_entity_labels = entity_encoder.transform(stripped_labels)
            for start, end in itertools.permutations(a.entities, 2):
                if (start.type, end.type) in allowed_entity_pairs:
                    rel = next((r.type for r in a.relations if
                                r.arg1 == start and r.arg2 == end
                                and r.type==relation_focus), 'none')


                    if start.start > end.start:  # This relation goes "backwards" in the text
                        start, end = end, start  # Reverse start and end tokens such that start occurs before end
                        reverse = 1
                    else:
                        reverse = 0

                    start_s, start_e = a.get_token_idxs(start)
                    end_s, end_e = a.get_token_idxs(end)

                    start_s, start_e = idx_mapping[start_s], idx_mapping[start_e]
                    end_s, end_e = idx_mapping[end_s], idx_mapping[end_e]

                    pre_window_start = max(0, start_s - window_pad)
                    post_window_end = min(len(tokens['input_ids'][0][:]), end_e + window_pad)


                    entity_encodings = entity_encoder.transform(
                        [start.type, end.type]).reshape(-1)

                    xs.append(
                        (
                            tokens['input_ids'][0][pre_window_start:post_window_end],
                            tokens['attention_mask'][0][pre_window_start:post_window_end],
                            onehot_entity_labels[pre_window_start:post_window_end],
                            entity_encodings,
                            reverse
                        )
                    )
                    ys.append(rel)

        self.x = xs
        self.y = relation_encoder.transform(ys)
        self.y_labels = ys

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x, y = self.x[idx], self.y[idx]
        tii = x[0].long()
        taa = x[1].long()
        el = torch.from_numpy(x[2]).long()
        ee = torch.from_numpy(x[3]).long()
        rev = torch.tensor(x[4])

        y = torch.tensor(y).long()

        return (tii, taa, el, ee, rev), y

    def collate_fn(self, sequences):
        xs, ys = zip(*sequences)
        tii, taa, el, ee, rev = tuple(zip(*xs))
        if self.cuda:
            return (rnn.pack_sequence(tii, False).cuda(),
                    rnn.pack_sequence(taa, False).cuda(),
                    rnn.pack_sequence(el, False).cuda(),
                    rnn.pack_sequence(ee, False).cuda(),
                    torch.tensor(rev).cuda()), torch.tensor(ys).long().cuda()
        else:
            return (rnn.pack_sequence(tii, False),
                    rnn.pack_sequence(taa, False),
                    rnn.pack_sequence(el, False),
                    rnn.pack_sequence(ee, False),
                    torch.tensor(rev)), torch.tensor(ys).long()



    def tokenize_and_align(self, word_array, ann_array):
        sentence_tokenized = self.tokenizer(' '.join(word_array), return_tensors='pt')
        sentence_labels = ([self.class_template["O"]] *
                  torch.sum(sentence_tokenized['attention_mask'],
                            dim=1).item())

        idx_mapping = dict()

        label_index = 0
        for word_index, (word, ann) in enumerate(zip(word_array, ann_array)):
            tokens = self.tokenizer.tokenize(word)
            for i, token in enumerate(tokens):
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                if label_index >= len(sentence_labels):
                    print(
                        "Warning: Labels are not fitting when mapping to tokenized versions of sentences.")
                    break
                while sentence_tokenized['input_ids'][0][label_index] != token_id:
                    label_index += 1
                if i == 0:
                    idx_mapping[word_index] = label_index
                    sentence_labels[label_index] = ann
                else:
                    sentence_labels[label_index] = ann.replace(
                        self.class_template["B"], self.class_template["I"])
                label_index += 1
        idx_mapping[len(word_array)] = sentence_tokenized['input_ids'].shape[1]

        return sentence_tokenized, sentence_labels, idx_mapping


