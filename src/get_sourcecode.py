import os
import re
import csv
import subprocess
import pandas as pd
from natsort import natsorted
from bs4 import BeautifulSoup

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


def query_source_classes(folder_path):
    """ query target directory of classes (relative to working directory) """
    query = 'defects4j export -p dir.src.classes'
    with subprocess.Popen(query, shell=True, executable='/bin/bash', cwd=folder_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as proc:
        output, _ = proc.communicate()
        source_classes = output.decode().split('\n')[-2]
        proc.wait()
        return source_classes


def extract_code_line(folder, filename, line_num):
    """ 在folder中查找file文件，并提取第line_num行中的代码"""
    for root, dirs, files in os.walk(folder):
        if filename in files:
            file_path = os.path.join(root, filename)
            try:
                with open(file_path) as f:
                    lines = f.readlines()
                    if isinstance(line_num, int):
                        try:
                            code_line = lines[line_num-1].strip()
                        except IndexError:
                            # Codec 17_buggy 18_buggy
                            print('IndexError: list index out of range')
                            return None
                    elif isinstance(line_num, list) and len(line_num) == 2:
                        code_line = '\n'.join(line.strip() for line in lines[line_num[0]-1:line_num[1]-1])
                    else:
                        raise ValueError('Invalid line_num format')
                    return code_line
            except UnicodeDecodeError:
                print("Failed to decode file: " + file_path)
                return None

def warning_location():
    csv_folder = f'{defects4j_dir}/csv'
    html_reports = f'{defects4j_dir}/spotbugs-html-reports'
    for pid in project_ids:
        log_dir = os.path.join(defects4j_dir, f'logs/{pid}')
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, f'{pid}-warning_location.log')
        log_file = open(log_file_path, 'w')
        csv_file = os.path.join(csv_folder, f'{pid}/{pid}-warnings.csv')
        warning_details = []
        source_codes = []
        folder_path = os.path.join(html_reports, f'{pid}')
        files = list_files_in_folder(folder_path)
        for file in files:
            bug_id = file[42:-21]
            work_folder = f'{defects4j_dir}/tmp/{bug_id}'
            source_classes = query_source_classes(work_folder)
            source_path = os.path.join(work_folder, source_classes)         
            with open(file, 'r') as f:
                report_content = f.read()
            soup = BeautifulSoup(report_content, 'html.parser')
            detail_tags = soup.find_all(text=re.compile(r"(click for details)"))
            cnt = 0
            for detail_tag in detail_tags:
                cnt += 1
                p_label = detail_tag.parent.parent
                warning_detail = p_label.get_text().strip() # 警报细节
                # print(warning_detail)
                warning_details.append(warning_detail)
                last_br_tag = p_label.find_all('br')[-1]
                next_sibling = last_br_tag.find_next_sibling(text=True).strip()
                # print(next_sibling)
                match1 = re.search(r'At\s+(\S+):(?:\[line\s+(\d+)\]|(?:\[lines\s+(\d+)-(\d+)\]))', next_sibling, re.IGNORECASE)
                # match1 can match r"Another occurrence at BooleanUtils.java:[line 376]"
                # r"At FormatCache.java:[line 246]" or # r"At ExtendedMessageFormat.java:[lines 378-389]"
                if match1: 
                    filename = match1.group(1)
                    if match1.group(2):
                        line_num = int(match1.group(2))
                    else:
                        line_num = [0,0]
                        line_num[0] = int(match1.group(3))
                        line_num[1] = int(match1.group(4))
                    code_line = extract_code_line(source_path, filename, line_num)
                    source_codes.append(code_line)
                    print(f'match1:{bug_id}-warning{cnt}...{code_line}')
                    log_file.write(f'\nmatch1:{bug_id}-warning{cnt}...{code_line}\n')
                else:
                    match2 = re.search(r'.*At\s+(\S+):\[line\s+(\d+)\]', warning_detail, re.IGNORECASE)
                    # some warning_details is not match, give them a opportunity. such as,
                    # r"At ExtendedMessageFormat.java:[line 1]Did you intend to override java.text.MessageFormat.equals(Object)"
                    if match2:
                        filename = match2.group(1)
                        line_num = int(match2.group(2))
                        # print('match2 is work!')
                        code_line = extract_code_line(source_path, filename, line_num)
                        source_codes.append(code_line)
                        print(f'match2:{bug_id}-warning{cnt}...{code_line}')
                        log_file.write(f'\nmatch2:{bug_id}-warning{cnt}...{code_line}\n')
                    else:
                        match3 = re.search(r'In class \S+ \S+ type (\S+)In (\S+)', warning_detail)
                        # Lang/40 r"In class org.apache.commons.lang.text.ExtendedMessageFormatField org.apache.commons.lang.text.ExtendedMessageFormat.registryActual
                        # type org.apache.commons.lang.text.FormatFactoryIn ExtendedMessageFormat.java"
                        if match3:
                            typename = match3.group(1)
                            filename = match3.group(2)
                            source_codes.append('NaN')
                            # 似乎FormatFactoryIn类存在问题，它被用于在ExtendedMessageFormat注册表中创建格式化对象的实例。
                            # 可能registryActual字段没有被正确初始化或引用了错误的对象，导致SE_BAD_FIELD bug的出现。
                            print(f'match3:{bug_id}-warning{cnt}...{filename}')
                            log_file.write(f'\nmatch3:{bug_id}-warning{cnt}...{filename}\n')
                        else:
                            # 算啦 填NaN吧毁灭吧
                            source_codes.append('NaN')
                            print('Damn it!!!!!!!!!!!!!!!!!!!!!')
                            log_file.write(f'\nNo match found for{bug_id}-warning{cnt}...\n')
                            
        df = pd.read_csv(csv_file)
        print(f'len(source_codes):{len(source_codes)}, len(warning_details):{len(warning_details)}')
        print(f'len(csv_file):{len(df)}')
        if len(source_codes) == len(df) and len(warning_details) == len(df):
            df = df.assign(**{'Source Code' : source_codes, 'Warning Detail': warning_details})
            df.to_csv(f'{csv_folder}/{pid}/{pid}-detailed-warnings.csv', index=False, quoting=csv.QUOTE_ALL)
        log_file.write(f'\nlen(source_codes):{len(source_codes)}, len(warning_details):{len(warning_details)}, len(csv_file):{len(df)}\n')
        log_file.write(f'\nGenerating {csv_folder}/{pid}/{pid}-detailed-warnings.csv\n')
        log_file.close()


if __name__ == '__main__':
    warning_location()    