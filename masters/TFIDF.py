import math
import os
import random
import pickle


class TFIDF(dict):
    def __init__(self, *args, **kwargs):
        self.word_document_count = dict()
        self.total_word_count = 0
        self.total_document_count = 0
        super(TFIDF, self).__init__(*args, **kwargs)
        # self.update_metrics()

    def update_metrics(self):
        self.total_word_count = sum(self.values())

    def save_dict_to_pickle(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_dict_from_pickle(file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data

    def tf(self, word):
        if self.total_word_count == 0:
            self.update_metrics()
        return self.get(word, 0)/self.total_word_count

    def idf(self, word):
        if self.total_word_count == 0:
            self.update_metrics()
        return math.log(self.total_document_count /
                        (self.word_document_count.get(word, 0) + 1))

    def tfidf(self, word):
        return self.tf(word) * self.idf(word)


if __name__ == "__main__":

    from utilities import StopWatch

    stopwatch = StopWatch(memory=True)

    from utilities.argparseactions import ArgumentParser, IterFilesAction, \
        FileAction

    parser = ArgumentParser(
        description='Output a TF-IDF pickle.')
    parser.add_argument('txtpath', action=IterFilesAction, recursive=True,
                        suffix='.txt',
                        help='Path to txt file or directory to parse.')
    parser.add_argument('outdir', action=FileAction, mustexist=False,
                        help='Output location for model files.')
    parser.add_argument('-v', "--verbose", action="store_true",
                        help="Verbose logs.")
    parser.add_argument('-n', '--num_files', help='Number of fies to take in',
                        type=int, default=-1)
    parser.add_argument('-M','--model',default='none',help='Base model to train from. Defaults to none and will not use a tokenizer.')


    command_args = parser.parse_args()

    print(f"Base model: {command_args.model}", flush=True)
    print(f"Output dir: {command_args.outdir}", flush=True)
    print(f"Number of files: {command_args.num_files}", flush=True)

    params = {}
    params['outdir'] = command_args.outdir
    params['verbose'] = command_args.verbose
    params['seed'] = 42  # for reproducibility
    params['base_model'] = command_args.model  # pretrained model

    print("\nBeginning to read files..\n", flush=True)
    file_paths = []
    for path in command_args.txtpath:
        file_paths.append(path)
    random.shuffle(file_paths)
    if command_args.num_files > 0:
        file_paths = file_paths[:command_args.num_files]
    out_directory = params['outdir']
    file_list_dir = os.path.join(out_directory,
                                 f"tfidf_{command_args.model}_file_list.txt")
    with open(file_list_dir, 'w') as f:
        for file in file_paths:
            f.write(file + "\n")
    stopwatch.tick("Finished parsing file list", report=True)

    print(f"\n\nWill be reading {len(file_paths)} files. Proceeding..\n\n",
          flush=True)

    tokenizer = None
    if command_args.model != 'none':
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(command_args.model)
        print(f"Tokenizer loaded from {command_args.model}", flush=True)

    tfidf = TFIDF()
    for nfile, path in enumerate(file_paths):
        if nfile % 10 == 0:
            print(f"Reading file {nfile+1}/{len(file_paths)}", flush=True)
        with open(path, 'r') as f:
            word_set = set()
            tfidf.total_document_count += 1
            for line in f:
                word_array = line.split()
                if tokenizer is not None:
                    word_array = tokenizer.tokenize(line)
                for word in word_array:
                    if word not in word_set:
                        tfidf.word_document_count[word] = tfidf.word_document_count.get(word, 0) + 1
                        word_set.add(word)
                    if word not in tfidf:
                        tfidf[word] = 0
                    tfidf[word] += 1

    tfidf.update_metrics()
    tfidf.save_dict_to_pickle(os.path.join(params['outdir'], f'tfidf_{command_args.model}_{len(file_paths)}.pkl'))

    stopwatch.tick('Finished making the TFIDF pickle', report=True)

    stopwatch.report()
