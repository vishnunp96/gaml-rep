from torch.utils.data import Dataset
import torch
''' 
Loads text files on demand and tokenizes them using an initialized tokenizer.
The text files are assumed to be:
- One sentence per line
- Each sentence pre-formatted to be max N tokens long with x overlapping tokens
'''
class BertFileOnDemandDataset(Dataset):
    def __init__(self, text_paths, tokenizer, max_length):
        self.text_paths = text_paths
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lookup = self.create_lookup()

    def create_lookup(self):
        # Create a list of tuples with the path and line index
        lookup = []
        for path in self.text_paths:
            with open(path, 'r') as f:
                for lineIndex, line in enumerate(f):
                    lookup.append((path, lineIndex))
        return lookup

    def __len__(self):
        return len(self.lookup)

    def __getitem__(self, idx):
        # lazy loading
        path, lineIndex = self.lookup[idx]
        with open(path, 'r') as f:
            for lineNumber, line in enumerate(f):
                if lineNumber == lineIndex:
                    text = line.strip()
                    break

        encoding = self.tokenizer(text,
                                  return_tensors='pt',
                                  max_length=self.max_length,
                                  padding='max_length',
                                  truncation=True)

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        # Create masked input and labels for masked language modeling
        masked_input, labels = self.mask_tokens(input_ids)

        return {
            'input_ids': masked_input,
            'attention_mask': attention_mask,
            'labels': labels
        }

    def mask_tokens(self, inputs):
        # The labels are the same as the inputs (self-supervised learning)
        # The inputs are randomly masked and the model has to predict the original tokens
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, 0.15)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # 80% of the time, replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(
            torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token)

        # 10% of the time, replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape,
                                                    0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape,
                                     dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) keep the masked input tokens unchanged

        return inputs, labels