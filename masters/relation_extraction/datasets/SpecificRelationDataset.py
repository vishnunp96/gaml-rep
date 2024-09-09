import numpy

from annotations.brattowindow import StandoffLabels
from annotations.datasets import AnnotationDataset
import itertools
import torch
import torch.nn.utils.rnn as rnn


# Modelled after RelationSplitSpansDirectedLabelledIndexesDataset
class SpecificRelationDataset(AnnotationDataset):
    def __init__(self, ann_list, token_indexes, fallback_index, labels,
                 entity_labels, relation_focus, allowed_entity_pairs,
                 window_pad, cuda=False):
        super(SpecificRelationDataset, self).__init__(ann_list, cuda)

        self.relation_focus = relation_focus
        self.allowed_entity_pairs = allowed_entity_pairs
        xs = []
        ys = []
        self.origins = []

        outside_label = entity_labels.transform([StandoffLabels.outside_label])
        for a in self.anns:
            stripped_labels = [l.split('_')[0] for l in a.labels]
            tokens, elabels = a.tokens, entity_labels.transform(stripped_labels)
            idxs = numpy.array(
                ([0] * window_pad) + [token_indexes.get(t, fallback_index) for t
                                      in tokens] + ([0] * window_pad))
            label_padding = entity_labels.transform(
                [StandoffLabels.outside_label] * window_pad)
            elabels = numpy.concatenate((label_padding, elabels, label_padding))
            for start, end in itertools.permutations(a.entities, 2):
                if (start.type, end.type) in allowed_entity_pairs:
                    rel = next((r.type for r in a.relations if
                                r.arg1 == start and r.arg2 == end
                                and r.type==relation_focus), 'none')

                    self.origins.append((a, start, end))

                    if start.start > end.start:  # This relation goes "backwards" in the text
                        start, end = end, start  # Reverse start and end tokens such that start occurs before end
                        reverse = 1
                    else:
                        reverse = 0

                    # print(f'Relation: {rel}')
                    # print(repr(start))
                    # print(repr(end))

                    start_s, start_e = a.get_token_idxs(start)
                    end_s, end_e = a.get_token_idxs(end)

                    start_s, start_e = start_s + window_pad, start_e + window_pad
                    end_s, end_e = end_s + window_pad, end_e + window_pad

                    pre_window = idxs[(start_s - window_pad):start_s]
                    pre_window_labels = elabels[(start_s - window_pad):start_s]
                    start_tokens = idxs[start_s:start_e]
                    start_tokens_labels = elabels[start_s:start_e]
                    between_tokens = numpy.concatenate(
                        ([0], idxs[start_e:end_s], [0]))
                    between_tokens_labels = numpy.concatenate(
                        (outside_label, elabels[start_e:end_s], outside_label))
                    end_tokens = idxs[end_s:end_e]
                    end_tokens_labels = elabels[end_s:end_e]
                    post_window = idxs[end_e:(end_e + window_pad)]
                    post_window_labels = elabels[end_e:(end_e + window_pad)]

                    entity_encodings = entity_labels.transform(
                        [start.type, end.type]).reshape(-1)

                    xs.append(
                        (
                            (pre_window, pre_window_labels),
                            (start_tokens, start_tokens_labels),
                            (between_tokens, between_tokens_labels),
                            (end_tokens, end_tokens_labels),
                            (post_window, post_window_labels),
                            entity_encodings,
                            reverse
                        )
                    )
                    ys.append(rel)

        self.x = xs
        self.y = labels.transform(ys)
        self.y_labels = ys

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x, y = self.x[idx], self.y[idx]

        x, y = tuple(
            (torch.from_numpy(ts).long(), torch.from_numpy(ls).long()) for
            ts, ls in x[:-2]) + (torch.from_numpy(x[-2]).float(),
                                 torch.tensor([x[-1]]).float()), torch.tensor(
            y).long()

        return x, y

    def collate_fn(self, sequences):
        xs, ys = zip(*sequences)
        unpack_xs = tuple(zip(*xs))
        unpacked_labels = [tuple(zip(*u)) for u in unpack_xs[:-2]]
        if self.cuda:
            return tuple((rnn.pack_sequence(ts, False).cuda(),
                          rnn.pack_sequence(ls, False).cuda()) for ts, ls in
                         unpacked_labels) + (
                   torch.stack(unpack_xs[-2], 0).cuda(),
                   torch.stack(unpack_xs[-1], 0).cuda()), torch.stack(ys,
                                                                      0).cuda()
        else:
            return tuple(
                (rnn.pack_sequence(ts, False), rnn.pack_sequence(ls, False)) for
                ts, ls in unpacked_labels) + (torch.stack(unpack_xs[-2], 0),
                                              torch.stack(unpack_xs[-1],
                                                          0)), torch.stack(ys,
                                                                           0)
