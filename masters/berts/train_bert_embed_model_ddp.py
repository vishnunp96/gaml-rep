import os.path

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import BertTokenizer, BertForMaskedLM, \
    get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import random
from BertFileOnDemandDataset import BertFileOnDemandDataset
from helpers import fprint, set_seed
from checkpointing import save_checkpoint


def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)


def fine_tune_with_loss(rank, params, text_data, world_size):
    # Initialize the parallel group
    ddp_setup(rank, world_size)

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
    model = BertForMaskedLM.from_pretrained(base_model,
                                            ignore_mismatched_sizes=True)

    fprint(f"\n\nUsing cuda?: {torch.cuda.device_count()}\n\n", rank)
    fprint(
        f"\n\nCUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}\n\n",
        rank)

    # 3. Wrap the model with DDP
    device = torch.device(
        f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = DDP(model, device_ids=[rank])
    print("Wrapped model with DDP.", flush=True)

    # 4. Create data loaders
    # List of sentences as input
    train_texts, val_texts = train_test_split(text_data, test_size=0.1,
                                              random_state=42)

    train_dataset = BertFileOnDemandDataset(train_texts, tokenizer,
                                            max_length=max_sentence_length)
    val_dataset = BertFileOnDemandDataset(val_texts, tokenizer,
                                          max_length=max_sentence_length)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size,
                                       rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size,
                                     rank=rank)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=data_loader_workers, pin_memory=True,
                              sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            num_workers=data_loader_workers,
                            pin_memory=True, sampler=val_sampler)
    fprint("Created data loaders.", rank)

    # 5. Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=total_steps)
    print(f"Set up optimizer and scheduler. Starting training..", flush=True)

    save_step = len(train_loader) // 10
    fprint(f"Will be saving model every {save_step} steps of training.", rank)
    # 6. Train and evaluate the model
    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        next_save = save_step
        fprint(f"Epoch {epoch + 1}/{num_epochs} training starts..", rank)
        for loop, batch in enumerate(train_loader):
            fprint(f"\tBatch {loop + 1}/{len(train_loader)}", rank)
            optimizer.zero_grad()

            fprint(f"\t\tLoading inputs..", rank, verbose)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            fprint(f"\t\tForward pass..", rank, verbose)
            outputs = model(input_ids, attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss
            fprint(f"\t\tBackward pass..", rank, verbose)
            loss.backward()
            fprint(f"\t\tOptimizer step..", rank, verbose)
            optimizer.step()
            fprint(f"\t\tScheduler step..", rank, verbose)
            scheduler.step()
            fprint(f"\t\tBatch complete.", rank, verbose)
            if rank == 0 and loop > next_save:
                next_save += save_step
                save_checkpoint(model, base_model, optimizer, scheduler, loop, epoch, loss,
                                model_save_path, prev_model_save_path)

        dist.barrier()
        # Validation
        model.eval()
        val_loss = 0
        fprint(f"Epoch {epoch + 1}/{num_epochs} validation starts..", rank)
        with torch.no_grad():
            for loop, batch in enumerate(val_loader):
                fprint(f"\tBatch {loop + 1}/{len(val_loader)}", rank)
                fprint(f"\t\tLoading inputs..", rank, verbose)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                fprint(f"\t\tForward pass..", rank, verbose)
                outputs = model(input_ids, attention_mask=attention_mask,
                                labels=labels)
                val_loss += outputs.loss.item()
                fprint(f"\t\tBatch complete.", rank, verbose)

        fprint(
            f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss / len(val_loader)}",
            rank)
        dist.barrier()
        if rank == 0:
            save_checkpoint(model, base_model, optimizer, scheduler, 0, epoch + 1, val_loss,
                            model_save_path, prev_model_save_path)
    fprint("Training complete.\n\n", rank)

    dist.destroy_process_group()


if __name__ == "__main__":

    from utilities import StopWatch

    stopwatch = StopWatch(memory=True)

    from utilities.argparseactions import ArgumentParser, IterFilesAction, \
        FileAction

    parser = ArgumentParser(
        description='Train BERT model on LateXML text data.')
    parser.add_argument('txtpath', action=IterFilesAction, recursive=True,
                        suffix='.txt',
                        help='Path to txt file or directory to parse.')
    parser.add_argument('outdir', action=FileAction, mustexist=False,
                        help='Output location for model files.')
    parser.add_argument('-v', "--verbose", action="store_true",
                        help="Verbose logs.")
    parser.add_argument('-n', '--num_files', help='Number of fies to take in',
                        type=int, default=-1)
    parser.add_argument('-M','--model',default='bert-base-cased',help='Base model to train from. Defaults to \'bert-base-cased\'.')


    command_args = parser.parse_args()
    world_size = torch.cuda.device_count()

    print(f"World size: {world_size}", flush=True)
    print(f"Base model: {command_args.model}", flush=True)
    print(f"Model dir: {command_args.outdir}", flush=True)
    print(f"Input dir: {command_args.txtpath}", flush=True)
    print(f"Number of files: {command_args.num_files}", flush=True)

    params = {}
    params['outdir'] = command_args.outdir
    params['verbose'] = command_args.verbose
    params['max_sentence_length'] = 256  # max: 512; has to be more than number of words per sentence in processed files
    params['sentence_overlap'] = 16  # same as above
    params['data_loader_workers'] = 2  # based on CPU RAM
    params['num_epochs'] = 2  # max 2 epochs
    params['batch_size'] = 40  # based on GPU RAM and max sentence length. 16GB GPU RAM can handle 128 * 80
    params['seed'] = 42  # for reproducibility
    params['base_model'] = command_args.model  # pretrained model
    stripped_model_name = command_args.model.split('/')[-1].split('-')[-1]
    params['model_name'] = f"{stripped_model_name}_{command_args.num_files}"

    print("\nBeginning to read files..\n", flush=True)
    text_data = []
    num_files = 0
    for path in command_args.txtpath:
        text_data.append(path)
    random.shuffle(text_data)
    if command_args.num_files > 0:
        text_data = text_data[:command_args.num_files]
    out_directory = params['outdir']
    file_list_dir = os.path.join(out_directory,
                                 f"{params['model_name']}_file_list.txt")
    with open(file_list_dir, 'w') as f:
        for file in text_data:
            f.write(file + "\n")
    stopwatch.tick("Finished deciding file list", report=True)

    print(f"\n\nWill be reading {len(text_data)} files. Proceeding..\n\n",
          flush=True)

    torch.multiprocessing.spawn(fine_tune_with_loss, args=(
        params, text_data, world_size),
                                nprocs=world_size,
                                join=True)

    stopwatch.tick('Finished training and evaluation', report=True)

    stopwatch.report()
