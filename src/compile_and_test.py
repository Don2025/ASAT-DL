import os
import natsort
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor

defects4j_dir = os.path.expanduser('~/defects4j')
project_ids = ['Chart', 'Cli', 'Closure', 'Codec', 'Collections', 'Compress', 'Csv', 'Gson', 'JacksonCore',
               'JacksonDatabind', 'JacksonXml', 'Jsoup', 'JxPath', 'Lang', 'Math', 'Mockito', 'Time']


def compile_and_test(pid, lock):
    cmd1 = 'defects4j compile'
    cmd2 = 'defects4j test'
    log_dir = os.path.join(defects4j_dir, f'logs/{pid}')
    os.makedirs(log_dir, exist_ok=True)
    project_path = os.path.join(defects4j_dir, f'tmp/{pid}')
    subfolders = natsort.natsorted(os.scandir(project_path), key=lambda x: x.name)
    for subfolder in subfolders:
        if subfolder.is_dir():
            log_file_path = os.path.join(log_dir, f'{subfolder.name}-test.log')
            lock.acquire()
            log_file = open(log_file_path, 'w')
            print(f'Compiling {subfolder.path}...')
            log_file.write(f'Compiling {subfolder.path}...\n')

            with subprocess.Popen(cmd1, shell=True, executable='/bin/bash', cwd=subfolder.path, stdout=subprocess.PIPE) as proc1:
                stdout_data, stderr_data = proc1.communicate()
                log_file.write(stdout_data.decode('utf-8')+'\n')
                
            print(f'Testing {subfolder.path}...')
            log_file.write(f'Testing {subfolder.path}...'+'\n')
            
            with subprocess.Popen(cmd2, shell=True, executable='/bin/bash', cwd=subfolder.path, stdout=subprocess.PIPE) as proc2:
                stdout_data, stderr_data = proc2.communicate()
                print(stdout_data.decode('utf-8'))
                log_file.write(stdout_data.decode('utf-8')+'\n')                
            log_file.close()
            lock.release()


def test_all_projects():
    """ Compiling and testing all projects """
    lock = threading.Lock()
    with ThreadPoolExecutor() as executor:
        for pid in project_ids:
            executor.submit(compile_and_test, pid, lock)


if __name__ == '__main__':
    test_all_projects()