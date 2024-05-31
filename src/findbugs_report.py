import os
import threading
import subprocess
from natsort import natsorted
from concurrent.futures import ThreadPoolExecutor

defects4j_dir = os.path.expanduser('~/defects4j')

def query_target_classes(folder_path):
    """ query target directory of classes (relative to working directory) """
    query = 'defects4j export -p dir.bin.classes'
    with subprocess.Popen(query, shell=True, executable='/bin/bash', cwd=folder_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as proc:
        output, _ = proc.communicate()
        target_classes = output.decode().split('\n')[-2]
        proc.wait()
        return target_classes

def findbugs_report(cmd, folder_path, log_file, lock):
    with subprocess.Popen(cmd, shell=True, executable='/bin/bash', cwd=folder_path, stdout=subprocess.PIPE) as proc:
        stdout_data, stderr_data = proc.communicate()
        thread_id = threading.current_thread().ident
        lock.acquire()
        with open(log_file, 'w') as f:
            f.write(f'Thread {thread_id}: {cmd}\n')
            f.write(f'Stdout: {stdout_data.decode()}\n')
            f.write(f'Stderr: {stderr_data.decode()}\n')
        lock.release()

def findbugs_all_projects(work_folder):
    work_folder = os.path.abspath(os.path.join(os.path.expanduser('~'), work_folder))
    print(work_folder)
    with ThreadPoolExecutor(max_workers=8) as executor:
        for root, dirs, files in os.walk(work_folder):
            # Traverse all buggy versions, 835 in total 
            if root.count(os.sep) == 5:
                for folder in natsorted(dirs):
                    if folder[-5:] == 'buggy':
                        folder_path = os.path.join(root, folder)
                        print(f'Processing {folder_path}...')
                        pid = os.path.basename(os.path.dirname(folder_path))
                        target_classes = query_target_classes(folder_path)
                        target_path = f'{pid}/{folder}/{target_classes}' # relative to work_folder
                        exclude_filter_path = os.path.join(folder_path, 'findbugs-exclude-filter.xml')
                        output_path = f'{defects4j_dir}/findbugs-html-reports/{pid}'
                        os.makedirs(output_path, exist_ok=True)
                        output_file = os.path.join(f'{output_path}', f'{folder}-findbugs_report.html')
                        os.system(f'touch {output_file}')
                        if os.path.exists(exclude_filter_path):
                            cmd = f'findbugs -html -output {output_file} -exclude {exclude_filter_path} {target_path}'
                        else:
                            cmd = f'findbugs -html -output {output_file} {target_path}'
                        log_dir = os.path.join(defects4j_dir, f'logs/{pid}')
                        os.makedirs(log_dir, exist_ok=True)
                        log_file_path = os.path.join(log_dir, f'{folder}-findbugs.log')
                        lock = threading.Lock()
                        executor.submit(findbugs_report, cmd, work_folder, log_file_path, lock)

       
if __name__ == '__main__':
   findbugs_all_projects(f'{defects4j_dir}/tmp')