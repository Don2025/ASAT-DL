import os
import csv
import pandas as pd
from natsort import natsorted

defects4j_dir = os.path.expanduser('~/defects4j')
project_ids = ['Chart', 'Cli', 'Closure', 'Codec', 'Collections', 'Compress', 'Csv', 'Gson', 'JacksonCore',
               'JacksonDatabind', 'JacksonXml', 'Jsoup', 'JxPath', 'Lang', 'Math', 'Mockito', 'Time']

def list_files_in_folder(folder_path):
    l = []
    files = os.listdir(folder_path)
    for file in natsorted(files):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            l.append(file_path)
    return l


def merge_csv(work_folder, fixed=False):
    # 305036 warnings in total
    # cnt = 0
    csv_folder = f'{defects4j_dir}/csv'
    for pid in project_ids:
        folder_path = os.path.join(csv_folder, f'{pid}')
        files = list_files_in_folder(folder_path)
        df_res = pd.DataFrame()
        for file in files:
            df = pd.read_csv(file)
            df_res = pd.concat([df_res, df], ignore_index=True)
        df_res.to_csv(f'{folder_path}/{pid}-warnings.csv', index=False, quoting=csv.QUOTE_ALL)
        print(f'Generating {folder_path}/{pid}-warnings.csv')
        # print(f'{pid} has {len(df_res)}')
        # cnt += len(df_res)
    # print(f'{cnt} in total.')


if __name__ == '__main__':
    work_folder = f'{defects4j_dir}/spotbugs-fixed-html-reports'
    merge_csv(work_folder, True)