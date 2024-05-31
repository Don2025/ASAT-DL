import os
import csv
import subprocess
import pandas as pd
from natsort import natsorted

defects4j_dir = os.path.expanduser('~/defects4j')
project_ids = ['Chart', 'Cli', 'Closure', 'Codec', 'Collections', 'Compress', 'Csv', 'Gson', 'JacksonCore',
               'JacksonDatabind', 'JacksonXml', 'Jsoup', 'JxPath', 'Lang', 'Math', 'Mockito', 'Time']

def query_modified_classes(folder_path):
    """ query target directory of classes (relative to working directory) """
    query = 'defects4j export -p classes.modified'
    with subprocess.Popen(query, shell=True, executable='/bin/bash', cwd=folder_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as proc:
        output, _ = proc.communicate()
        modified_classes = output.decode().split('\n')[-2]
        proc.wait()
        return modified_classes


def list_modified_classes(work_folder):
    work_folder = os.path.abspath(os.path.join(os.path.expanduser('~'), work_folder))
    print(work_folder)
    for root, dirs, files in os.walk(work_folder):
        # Traverse all buggy versions, 835 in total 
        dict_df = {}
        if root.count(os.sep) == 5:
            for folder in natsorted(dirs):
                if folder[-5:] == 'buggy':
                    folder_path = os.path.join(root, folder)
                    print(f'Processing {folder_path}...')
                    pid = os.path.basename(os.path.dirname(folder_path))
                    modified_classes = query_modified_classes(folder_path)
                    print(f'{pid}/{folder}...{modified_classes}')
                    dict_df[f'{pid}/{folder}'] = modified_classes

            # Save the dictionary as CSV
            csv_file = f'{defects4j_dir}/csv/{pid}/{pid}-modified_classes.csv'
            # df = pd.DataFrame(dict_df.items(), columns=['Bug Id', 'Modified Classes'])
            # df.to_csv(csv_file, index=False)
            # Equivalent to the following code
            with open(csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Bug Id', 'Modified Classes'])
                for key, value in dict_df.items():
                    writer.writerow([key, value])

            print(f'Saved as {csv_file}')


def label_marking0():
    csv_folder = f'{defects4j_dir}/csv'
    for pid in project_ids:
        print(f'Processing {csv_folder}/{pid}...')
        df = pd.read_csv(f'{csv_folder}/{pid}/{pid}-modified_classes.csv')
        modified_classes = df['Modified Classes'].tolist()
        tokens = [x.split('.')[-1] for x in modified_classes]
        df = pd.read_csv(f'{csv_folder}/{pid}/{pid}-warnings.csv')
        warnings = set(df['Warning'])
        warnings_df = pd.DataFrame(list(warnings), columns=['Warning'])
        warnings_df['Label'] = warnings_df['Warning'].str.contains('|'.join(tokens)).astype(int)
        warnings_df.to_csv(f'{csv_folder}/{pid}/{pid}-warnings_unique.csv', index=False, quoting=csv.QUOTE_ALL)
   
def label_marking():
    csv_folder = f'{defects4j_dir}/csv'
    cnt = 0
    for pid in project_ids:
        print(f'Processing {csv_folder}/{pid}...')
        df = pd.read_csv(f'{csv_folder}/{pid}/{pid}-modified_classes.csv')
        modified_classes = df['Modified Classes'].tolist()
        tokens = [x.split('.')[-1] for x in modified_classes]
        df = pd.read_csv(f'{csv_folder}/{pid}/{pid}-detailed-warnings.csv')
        warnings = set(zip(df['Warning'].tolist(), df['Source Code'].tolist(), df['Warning Detail'].tolist()))
        warnings_df = pd.DataFrame(list(warnings), columns=['Warning', 'Source Code', 'Warning Detail'])
        warnings_df['Label'] = warnings_df['Warning'].str.contains('|'.join(tokens)).astype(int)
        print(len(warnings_df))
        cnt += len(df)
        warnings_df.to_csv(f'{csv_folder}/{pid}/{pid}-warnings_soha.csv', index=False, quoting=csv.QUOTE_ALL)     
    print(cnt)

if __name__ == '__main__':
    # list_modified_classes(f'{defects4j_dir}/tmp')
    label_marking()