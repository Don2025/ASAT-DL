from bs4 import BeautifulSoup
import pandas as pd

# def parse_findbugs_html(html_file):
#     with open(html_file, 'r') as f:
#         html = f.read()
#     soup = BeautifulSoup(html, 'html.parser')
#     tables = soup.find_all('table', {"class": 'warningtable'})
#     # Extract H2 tags and table rows
#     h2_tag_texts = [h2.text.strip() for h2 in soup.find_all("h2")]
#     #table_row_texts = [row.text.strip() for row in table_rows]

#     # Print the extracted H2 tag text and table row text
#     print("H2 Tags:")
#     for h2_text in h2_tag_texts:
#         print("- " + h2_text)

#     # print("\nTable Rows:")
#     # for row_text in table_row_texts:
#     #     print("- " + row_text)


if __name__ == '__main__':
    html_file = '/home/tyd/defects4j/findbugs-html-reports/Lang/1_buggy-findbugs_report.html'
    # parse_findbugs_html(html_file)
    df = pd.read_html(html_file)
    print(df[2])