import pandas as pd
import os


def find_files(directory, ending):
    matching_files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(ending):
                matching_files.append(os.path.join(root, filename))
    return matching_files


def process_csv_files(csv_files, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_list = []
    for idx, csv_file in enumerate(csv_files):
        folder_name = os.path.basename(os.path.dirname(csv_file))
        # Read the CSV file
        df = pd.read_csv(csv_file)

        # Add a new column with a unique value
        df['Folder Name'] = folder_name
        df_list.append(df)


    # Concatenate all the dataframes
    df = pd.concat(df_list)
    output_file = os.path.join(output_dir, 'all_class_metrics.csv')
    df.to_csv(output_file, index=False)


# Example usage
csv_files = find_files('/vol/bitbucket/vnp23/model_attempts/bert_lstm_ner/tuning/models', 'class_metrics.csv')

process_csv_files(csv_files, '/vol/bitbucket/vnp23/model_attempts/bert_lstm_ner/tuning/models')