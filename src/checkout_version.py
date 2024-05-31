import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

defects4j_dir = os.path.expanduser('~/defects4j')
project_ids = ['Chart', 'Cli', 'Closure', 'Codec', 'Collections', 'Compress', 'Csv', 'Gson', 'JacksonCore', 'JacksonDatabind', 'JacksonXml', 'Jsoup', 'JxPath', 'Lang', 'Math', 'Mockito', 'Time']

def checkout_version(pid, vid, fixed=False):
    """ Checkout a buggy/fixed source code version """
    version = 'fixed' if fixed else 'buggy' 
    work_dir = os.path.join(defects4j_dir, f'tmp/{pid}/{vid}_{version}')
    os.makedirs(work_dir, exist_ok=True)
    log_dir = os.path.join(defects4j_dir, f'logs/{pid}')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f'{vid}_{version}-checkout.log')
    cmd = f'defects4j checkout -p {pid} -v {vid}{version[0]} -w {work_dir}'
    try:
        with open(log_file_path, 'w') as log_file:
            process  = subprocess.Popen(cmd, shell=True, executable='/bin/bash', cwd=defects4j_dir, stdout=log_file, stderr=subprocess.STDOUT)
            process.wait()
        print(f'Checkout {pid}-{vid}_{version} success.')
    except subprocess.CalledProcessError as e:
        print(f'Error occurred when checking out {pid}-{vid}_{version}: {e.output}')


def checkout_all_projects():
    """ Before checking out all projects, ensure that 
        there is approximately 20GB of free space to generate a working directory.
        Chart need to use svn, sudo apt install subversion. 
        """
    bugs_number = {'Chart': 26, 'Cli': 40, 'Closure': 176, 'Codec': 18, 'Collections': 28, 'Compress': 47, 'Csv': 16, 
                  'Gson': 18, 'JacksonCore': 26, 'JacksonDatabind': 112, 'JacksonXml': 6, 'Jsoup': 93, 'JxPath': 22, 
                  'Lang': 65, 'Math': 106, 'Mockito': 38, 'Time': 27}
    with ThreadPoolExecutor() as executor:
        for pid in project_ids:
            vid_cnt = bugs_number.get(pid, 0)
            executor.map(checkout_version, [pid]*vid_cnt, range(1, vid_cnt+1))
            executor.map(checkout_version, [pid]*vid_cnt, range(1, vid_cnt+1), [True]*vid_cnt)

if __name__ == '__main__':
    checkout_all_projects()