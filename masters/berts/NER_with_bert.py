import os.path

import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from transformers import BertTokenizer, BertForTokenClassification
from sklearn.model_selection import train_test_split
import random
from annotations.annmlutils import open_anns
from helpers import fprint, set_seed
from checkpointing import save_checkpoint
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder


''' 
Loads text files on demand and tokenizes them using an initialized tokenizer.
The text files are assumed to be:
- One sentence per line
- Each sentence pre-formatted to be max N tokens long with x overlapping tokens
'''
class BertNERDataset(Dataset):
    def __init__(self, standoff_list, tokenizer, tokenizer_max_length,
                 sentence_split_params,
                 label_encoder,
                 class_template=None):
        self.class_template = class_template
        if self.class_template is None:
            self.class_template = {"O": "outside", "B": "begin", "I": "inside"}
        self.standoff_list = standoff_list
        self.tokenizer = tokenizer
        self.tokenizer_max_length = tokenizer_max_length
        self.sentence_split_length, self.sentence_overlap = sentence_split_params
        self.label_encoder = label_encoder

        words, anns, classes = self.load_annotations(standoff_list)
        self.sentence_tensors, self.labels = self.tokenize_annotations(
            words,
            anns)

    def load_annotations(self, standoff_list):
        # need to use standoffs instead of directory.
        # need to split appropriately as well.
        words = []
        annotations = []
        classes = set()

        for standoff in standoff_list:
            for i in range(0, len(standoff.tokens),
                           self.sentence_split_length - self.sentence_overlap):
                words.append(standoff.tokens[i:i + self.sentence_split_length])
                annotations.append(
                    standoff.labels[i:i + self.sentence_split_length])
                classes.update(annotations[-1])

        return words, annotations, classes

    def tokenize_annotations(self, words, annotations):
        sentence_tensors = []
        label_list = []
        for word_line, ann_line in zip(words, annotations):
            # get tokens
            sentence = " ".join(word_line)
            sentence_tensor = self.tokenizer(sentence, return_tensors='pt',
                                              max_length=self.tokenizer_max_length,
                                              padding='max_length',
                                              truncation=True)
            labels = ([self.class_template["O"]] *
                      torch.sum(sentence_tensor['attention_mask'],
                                dim=1).item())
            labels = ([self.class_template["O"]] *
                      self.tokenizer_max_length)


            label_index = 0
            for word, ann in zip(word_line, ann_line):
                tokens = self.tokenizer.tokenize(word)
                for i, token in enumerate(tokens):
                    token_id = self.tokenizer.convert_tokens_to_ids(token)
                    if label_index >= len(labels):
                        print("Warning: Labels are not fitting when mapping to tokenized versions of sentences.")
                        break
                    while sentence_tensor['input_ids'][0][
                        label_index] != token_id:
                        label_index += 1
                    if i == 0:
                        labels[label_index] = ann
                    else:
                        labels[label_index] = ann.replace(
                            self.class_template["B"], self.class_template["I"])
                    # labels[label_index] = ann.replace("begin", "inside")
                    label_index += 1
            sentence_tensors.append(sentence_tensor)
            label_list.append(self.label_encoder.transform(labels))

        return sentence_tensors, label_list

    def __len__(self):
        return len(self.sentence_tensors)

    def __getitem__(self, idx):
        sentence_tensor = self.sentence_tensors[idx]
        labels = self.labels[idx]

        input_ids = sentence_tensor['input_ids'].squeeze()
        token_type_ids = sentence_tensor['token_type_ids'].squeeze()
        attention_mask = sentence_tensor['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': labels
        }

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)


def fine_tune_with_loss(rank, params, ann_data, output_labels, world_size):
    # Initialize the parallel group
    # ddp_setup(rank, world_size)

    # 1. Load parameters
    verbose = params['verbose']
    num_epochs = params['num_epochs']
    max_sentence_length = params['max_sentence_length']
    batch_size = params['batch_size']
    set_seed(random, torch, params['seed'] + rank)
    data_loader_workers = params['data_loader_workers']
    fprint("Params received: " + str(params), rank, verbose)

    model_save_dir = params['outdir']
    model_save_path = os.path.join(model_save_dir, f"{params['model_name']}.pt")
    prev_model_save_path = os.path.join(model_save_dir,
                                        f"{params['model_name']}_prev.pt")
    fprint(f"Model will be saved to: {model_save_path}", rank)
    fprint(f"Prev model will be moved to: {prev_model_save_path}", rank,
           verbose)

    # 2. Load the model and tokenizer
    base_model = params['base_model']
    tokenizer = BertTokenizer.from_pretrained(base_model)
    model = BertForTokenClassification.from_pretrained(base_model,
                                                       num_labels=len(output_labels),
                                            ignore_mismatched_sizes=True)

    fprint(f"\n\nUsing cuda?: {torch.cuda.device_count()}\n\n", rank)
    fprint(
        f"\n\nCUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}\n\n",
        rank)

    # 3. Wrap the model with DDP
    device = torch.device(
        f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # model = DDP(model, device_ids=[rank])
    print("Wrapped model with DDP.", flush=True)
    label_encoder = LabelEncoder().fit(output_labels)

    # 4. Create data loaders
    # List of sentences as input
    train_texts, val_texts = train_test_split(ann_data, test_size=0.1,
                                              random_state=42)

    train_dataset = BertNERDataset(train_texts, tokenizer,
                                   tokenizer_max_length=max_sentence_length,
                                   sentence_split_params=(128, 16),
                                   label_encoder=label_encoder)
    val_dataset = BertNERDataset(val_texts, tokenizer,
                                 tokenizer_max_length=max_sentence_length,
                                 sentence_split_params=(128, 16),
                                 label_encoder=label_encoder)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=data_loader_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            num_workers=data_loader_workers,
                            pin_memory=True)
    fprint("Created data loaders.", rank)

    # 5. Set up optimizer and scheduler
    total_steps = len(train_loader) * num_epochs

    save_step = len(train_loader)
    fprint(f"Will be saving model every {save_step} steps of training.", rank)


    # NER SPECIFIC
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    print(f"Set up optimizer and scheduler. Starting training..", flush=True)
    # 6. Train and evaluate the model
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        # train_sampler.set_epoch(epoch)
        next_save = save_step
        fprint(f"Epoch {epoch + 1}/{num_epochs} training starts..", rank)
        for loop, batch in enumerate(train_loader):
            fprint(f"\tBatch {loop + 1}/{len(train_loader)}", rank)
            optimizer.zero_grad()

            fprint(f"\t\tLoading inputs..", rank, verbose)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            fprint(f"\t\tForward pass..", rank, verbose)
            outputs = model(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
            # loss = outputs.loss
            # print(len(labels), labels.shape, labels)
            # print("**************************\n\n")
            # print(len(outputs.logits), outputs.logits.shape, outputs.logits)
            loss = criterion(outputs.logits.view(-1, len(output_labels)), labels.view(-1))
            fprint(f"\t\tBackward pass..", rank, verbose)
            loss.backward()
            fprint(f"\t\tOptimizer step..", rank, verbose)
            optimizer.step()
            fprint(f"\t\tScheduler step..", rank, verbose)
            # scheduler.step()
            train_loss += loss.item()
            fprint(f"\t\tBatch complete. Training loss: {train_loss}", rank, verbose)
            if rank == 0 and loop > next_save:
                next_save += save_step
                save_checkpoint(model, base_model, optimizer, None, loop, epoch, loss,
                                model_save_path, prev_model_save_path)

        # dist.barrier()
        # Validation
        model.eval()
        val_loss = 0
        val_preds, val_true = [], []
        fprint(f"Epoch {epoch + 1}/{num_epochs} validation starts..", rank)
        with torch.no_grad():
            for loop, batch in enumerate(val_loader):
                fprint(f"\tBatch {loop + 1}/{len(val_loader)}", rank)
                fprint(f"\t\tLoading inputs..", rank, verbose)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['labels']

                fprint(f"\t\tForward pass..", rank, verbose)
                # outputs = model(input_ids, attention_mask=attention_mask,
                #                 labels=labels)
                # val_loss += outputs.loss.item()
                outputs = model(input_ids, attention_mask, token_type_ids)
                val_preds.extend(outputs.logits.argmax(dim=-1).tolist()[0])
                val_true.extend(labels.tolist()[0])
                fprint(f"\t\tBatch complete.", rank, verbose)

        val_f1 = f1_score(val_true, val_preds, average='macro')
        val_precision = precision_score(val_true, val_preds, average='macro', zero_division=0)
        val_recall = recall_score(val_true, val_preds, average='macro')
        fprint(
            f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss / len(val_loader)}",
            rank)
        fprint(f'Validation F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}', rank)
        # dist.barrier()

    fprint("Training complete.\n\n", rank)

    # dist.destroy_process_group()


if __name__ == "__main__":

    from utilities import StopWatch

    stopwatch = StopWatch(memory=True)

    from utilities.argparseactions import ArgumentParser, IterFilesAction, \
        FileAction

    def parse_tuple(s):
        return tuple(int(i) for i in s.split('-'))

    parser = ArgumentParser(
        description='Train BERT model on LateXML text data.')
    parser.add_argument('ann',action=IterFilesAction,recursive=True,suffix='.ann',help='Annotation file or directory containing files (searched recursively).')
    parser.add_argument('outdir', action=FileAction, mustexist=False,
                        help='Output location for model files.')
    parser.add_argument('-v', "--verbose", action="store_true",
                        help="Verbose logs.")
    parser.add_argument('-n', '--num_files', help='Number of fies to take in',
                        type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42, help='Random number seed for this training run.')
    parser.add_argument('--split', type=parse_tuple, default=(0.8,0.1,0.1), help='Data split for train-test-dev as hyphen-separated list, e.g. 60-20-20.')


    command_args = parser.parse_args()
    world_size = torch.cuda.device_count()

    params = {}
    params['outdir'] = command_args.outdir
    params['verbose'] = command_args.verbose
    params['max_sentence_length'] = 512  # max: 512; has to be more than number of words per sentence in processed files
    params['sentence_overlap'] = 16  # same as above
    params['data_loader_workers'] = 1  # based on CPU RAM
    params['num_epochs'] = 1  # max 2 epochs
    params['batch_size'] = 40  # based on GPU RAM and max sentence length. 16GB GPU RAM can handle 128 * 80
    params['seed'] = command_args.seed  # for reproducibility
    params['base_model'] = 'bert-base-cased'  # pretrained model
    params['model_name'] = f"bert_embedding_{command_args.num_files}"

    print("\nBeginning to read files..\n", flush=True)


    types = ['MeasuredValue','Constraint','ParameterSymbol','ParameterName','ConfidenceLimit','ObjectName','Definition']
    anns = open_anns(command_args.ann, types=types, use_labelled=True)
    output_labels = list(set(l for a in anns for t, l in a))
    print(len(output_labels), output_labels)

    fine_tune_with_loss(0,params,anns,output_labels,world_size)

    stopwatch.tick('Finished training and evaluation', report=True)

    stopwatch.report()
