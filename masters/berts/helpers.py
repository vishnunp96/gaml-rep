import numpy as np

def fprint(text, rank, verbose=True):
    if rank == 0 and verbose:
        print(text, flush=True)


def set_seed(random, torch, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_sentences(text, max_length, overlap):
    tokens = text.split()
    chunks = []
    for i in range(0, len(tokens), max_length - overlap):
        chunks.append(' '.join(tokens[i:i + max_length]))
    return chunks
