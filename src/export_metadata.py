import os
import csv
import subprocess
from concurrent.futures import ThreadPoolExecutor

defects4j_dir = os.path.expanduser('~/defects4j')
project_ids = ['Chart', 'Cli', 'Closure', 'Codec', 'Collections', 'Compress', 'Csv', 'Gson', 'JacksonCore',
               'JacksonDatabind', 'JacksonXml', 'Jsoup', 'JxPath', 'Lang', 'Math', 'Mockito', 'Time']


def project_metadata(pid, field_list):
    output_dir = os.path.join(defects4j_dir, f'data/{pid}')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{pid}-metadata.csv')
    field_str = ','.join(field_list)
    query = f'defects4j query -p {pid} -q "{field_str}" -o {output_file}'
    process = subprocess.Popen(query, shell=True, executable='/bin/bash', cwd=defects4j_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    process.wait()
    with open(output_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(field_list)
        writer.writerows(rows)
    
field_list = ['bug.id', 'project.id', 'project.name', 'project.vcs', 'project.repository', 'revision.id.buggy', 'revision.id.fixed', 'report.id', 'report.url']
def export_all_projects_metadata():
    """ Show metadata about all projects """
    with ThreadPoolExecutor() as executor:
        executor.map(project_metadata, project_ids, [field_list]*len(project_ids))

if __name__ == '__main__':
    export_all_projects_metadata()