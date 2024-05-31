import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

defects4j_dir = os.path.expanduser('~/defects4j')
project_ids = ['Chart', 'Cli', 'Closure', 'Codec', 'Collections', 'Compress', 'Csv', 'Gson', 'JacksonCore',
               'JacksonDatabind', 'JacksonXml', 'Jsoup', 'JxPath', 'Lang', 'Math', 'Mockito', 'Time']


def project_info(pid, save=False):
    """ Get information for a specific project """
    query = f'defects4j info -p {pid}'
    if save:
        log_dir = os.path.join(defects4j_dir, f'logs/{pid}')
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, f'{pid}-Info.log')
        with open(log_file_path, 'w') as log_file:
            process = subprocess.Popen(query, shell=True, executable='/bin/bash', cwd=defects4j_dir, stdout=log_file, stderr=subprocess.STDOUT)
        process.wait()
    else:
        process = subprocess.Popen(query, shell=True, executable='/bin/bash', cwd=defects4j_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        with process.stdout:
            output = process.stdout.read().decode()
            return pid, output
        process.wait()


def log_all_projects_info():
    """ Log information about all projects """
    with ThreadPoolExecutor() as executor:
        executor.map(project_info, project_ids, [True] * len(project_ids))


def show_all_projects_info():
    """ Just show information about all projects """
    with ThreadPoolExecutor() as executor:
        results = executor.map(project_info, project_ids)
        for res in results:
            print(f'{res[0]}, {res[1]}')
    

def bug_info(pid, vid):
    query = f'defects4j info -p {pid} -b {vid}'
    log_dir = os.path.join(defects4j_dir, f'logs/{pid}')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f'{vid}_bug-info.log')
    with open(log_file_path, 'w') as log_file:
        process = subprocess.Popen(query, shell=True, executable='/bin/bash', cwd=defects4j_dir, stdout=log_file, stderr=subprocess.STDOUT)
        process.wait()


def log_all_bugs_info():
    """ Log information about all bugs """
    bugs_number = {'Chart': 26, 'Cli': 40, 'Closure': 176, 'Codec': 18, 'Collections': 28, 'Compress': 47, 'Csv': 16, 
                  'Gson': 18, 'JacksonCore': 26, 'JacksonDatabind': 112, 'JacksonXml': 6, 'Jsoup': 93, 'JxPath': 22, 
                  'Lang': 65, 'Math': 106, 'Mockito': 38, 'Time': 27}
    with ThreadPoolExecutor() as executor:
        for pid in project_ids:
            vid_cnt = bugs_number.get(pid, 0)
            executor.map(bug_info, [pid]*vid_cnt, range(1, vid_cnt+1))


if __name__ == '__main__':
    # show_all_projects_info()
    # log_all_projects_info()
    log_all_bugs_info()