from bs4 import BeautifulSoup

def parse_findbugs_xml(xml_file):
    with open(xml_file, 'r') as f:
        xml_string = f.read()
    soup = BeautifulSoup(xml_string, 'xml')
    bug_instances = soup.find_all('BugInstance')
    result = []
    for bug_instance in bug_instances:
        class_name = bug_instance.find('Class')['classname'] if bug_instance.find('Class') is not None else None
        method_name = bug_instance.find('Method')['name'] if bug_instance.find('Method') is not None else None
        field_name = bug_instance.find('Field')['name'] if bug_instance.find('Field') is not None else None
        bug_pattern = bug_instance['type'] if 'type' in bug_instance.attrs else None
        bug_code = bug_instance['code'] if 'code' in bug_instance.attrs else None
        bug_category = bug_instance['category'] if 'category' in bug_instance.attrs else None
        sourceline = bug_instance.find('SourceLine')
        startline, endline = -1, -1
        if sourceline is not None:
            if 'start' in sourceline.attrs: 
                startline = sourceline['start']
            if 'end' in sourceline.attrs:    
                endline = sourceline['end']
        
        result.append(((bug_pattern, bug_code, bug_category), class_name[:class_name.rindex('.')] if '.' in class_name else None, class_name, field_name, method_name, (startline, endline)))
    return result


if __name__ == '__main__':
    # 暂且搁置解析xml脚本
    xml_file = '/home/tyd/defects4j/findbugs-xml-reports/Lang/1_buggy-findbugs_report.xml'
    alerts_xml = parse_findbugs_xml(xml_file)
    