import os.path

from utilities.parallel import parallel_results




def split_sentences(text, max_length, overlap):
    tokens = text.split()
    chunks = []
    for i in range(0, len(tokens), max_length - overlap):
        chunks.append(' '.join(tokens[i:i + max_length]))
    return chunks


def dowrite(path, results, outdir, sentlen, overlap):
    filename = os.path.basename(path)
    out_path = os.path.join(outdir, changeext(filename, '.txt'))

    with open(path, 'r') as f, open(out_path, 'w') as out:
        print("Writing to --> ", out_path)
        for line in f:
            stripped_line = line.strip()
            if not stripped_line:
                continue
            if len(stripped_line.split()) > args.sentlen:
                for sentence in split_sentences(stripped_line, sentlen,
                                                overlap):
                    out.write(sentence + '\n')
            else:
                out.write(stripped_line + '\n')

    results['files'] += 1


if __name__ == "__main__":

    from utilities.argparseactions import ArgumentParser,IterFilesAction, FileAction
    from utilities.fileutilities import changeext

    parser = ArgumentParser(description='Sample .txt latexml files, splitting into smaller sentences with overlap.')
    parser.add_argument('indir',action=IterFilesAction, recursive=True, suffix='.txt', help='Path to directory with input text files.')
    parser.add_argument('outdir',action=FileAction, mustexist=False, help='Output location for text files.')
    parser.add_argument('-s', '--sentlen', help='length of sentence in tokens', type=int, default=512)
    parser.add_argument('-o', '--overlap', help='number of tokens overlap', type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="Whether to run a trial version.")
    args = parser.parse_args()

    num_files = 0
    print(f"Indir:\t\t\t{args.indir}")
    print(f"Outdir:\t\t\t{args.outdir}")
    print(f"Sentence length:\t{args.sentlen}")
    print(f"Sentence overlap:\t{args.overlap}")
    print(f"Quick??:\t\t\t{args.quick}")
    print("Starting to read files..")

    results = parallel_results(dowrite, args.indir, additional_args=(args.outdir, args.sentlen, args.overlap), chunksize=5000, processes=16)
    print(results)

    print("Number of files written: ", num_files, flush=True)
