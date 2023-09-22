# Import the necessary packages
import pdfplumber
import re
import os

# reduces row to required elements
def fetch_row(row):
    res = []
    for element in row[:20]:
        res.append(element)

    return res


def read_both():  # returns both zentralzylinder_pattern, final_pattern_list

    # Read all the pdf file in the data directory
    data_location = "Data/"
    files_location = []

    for file in os.listdir(data_location):
        if file.endswith(".pdf"):
            files_location.append(os.path.join(data_location, file))

    for file in files_location:
        print(file)
        with pdfplumber.open(file) as pdf:
            pages = pdf.pages
            zentral_list = []
            zentral_codes_list = []
            schlussel_list = []
            schlussel_indices = []
            norm_list = []
            zentral_data = []
            zentral_codes_data = []
            schlussel_data = []
            norm_data = []
            zentral_middle = []
            normal_middle = []
            lookup_table = {}
            for ind, pdf_page in enumerate(pages):
                # extracts single page text
                print(f"Reading page {ind}")
                single_page_text = pdf_page.extract_text()

                if ind == 0:
                    code = single_page_text[: single_page_text.find(" ")]
                    date = single_page_text[single_page_text.find("Datum:") + 7 :]

                if "Referenzliste Zentralzylinder-Zentralcodes" in single_page_text:
                    contents = single_page_text.split("\n")
                    start = contents.index("Zentralzylinder Zentralcode")
                    contents = contents[start + 1 :]
                    current = contents[0][0]
                    for value in contents:
                        row = value.split(" ")
                        if row[0].isdigit():
                            value = value[value.find(" ") + 1 :]
                            current = row[0]
                            lookup_table[current] = ""
                            lookup_table[current] += value
                        else:
                            lookup_table[current] += "\n" + value

                # extracts Zentralzylinder
                elif "Zentralzylinder" in single_page_text:
                    table = pdf_page.extract_table()
                    if not table:
                        continue
                    flag = True
                    temp_row = []

                    for index, row in enumerate(table):
                        if index % 3 == 0 and row[0]:
                            zentral_middle.append(row[2])
                        if index % 3 == 0 and temp_row != []:
                            zentral_data.append(temp_row)
                            temp_row = []
                        if index % 3 == 0 and row[0]:
                            zentral_list.append(row[0][: row[0].find("\n")])

                        temp_row.append(fetch_row(row[5:]))
                    if len(temp_row) == 3:
                        zentral_data.append(temp_row)

                # extracts Zentralcodes
                elif "Zentralcodes" in single_page_text:
                    table = pdf_page.extract_table()
                    flag = True
                    temp_row = []
                    for index, row in enumerate(table):
                        if index % 3 == 0 and temp_row != []:
                            zentral_codes_data.append(temp_row)
                            temp_row = []
                        if index % 3 == 0 and row[0]:
                            zentral_codes_list.append(row[0])
                        temp_row.append(fetch_row(row[5:]))
                    if len(temp_row) == 3:
                        zentral_codes_data.append(temp_row)

                # extracts Normalzylinder
                elif "Normalzylinder" in single_page_text:
                    table = pdf_page.extract_table()
                    flag = True
                    temp_row = []
                    for index, row in enumerate(table):
                        if index % 3 == 0 and row[0]:
                            normal_middle.append(row[2])
                        if index % 3 == 0 and temp_row != []:
                            norm_data.append(temp_row)
                            temp_row = []
                        if index % 3 == 0 and row[0]:
                            if "\n" in row[0]:
                                norm_list.append(row[0][: row[0].find("\n")])
                            else:
                                norm_list.append(row[0])
                        temp_row.append(fetch_row(row[5:]))
                    if len(temp_row) == 3:
                        norm_data.append(temp_row)

                # extracts Schlüssel
                elif "Schlüssel" in single_page_text:
                    table = pdf_page.extract_table()
                    temp_row = []
                    for index, row in enumerate(table):
                        # print(index,row)
                        if index % 3 == 0 and temp_row != []:
                            schlussel_data.append(temp_row)
                            temp_row = []
                        if index % 3 == 0 and row[0]:
                            schlussel_list.append(row[0])
                            try:
                                schlussel_indices.append(row[1][1])
                            except IndexError:
                                schlussel_indices.append(row[1])

                        temp_row.append(fetch_row(row[5:]))
                    if len(temp_row) == 3:
                        schlussel_data.append(temp_row)

            zentral_list = [z for z in zentral_list if z]
            zentral_codes_list = [z for z in zentral_codes_list if z]
            zentral_data = zentral_data[: len(zentral_list)]

        # for index, value in enumerate(norm_list):
        #     print(norm_list[index], norm_data[index])

        # returns all lists in one dict
        final_data = {
            "zentralzylinder": (zentral_list, zentral_data),
            "normalzylinder": (norm_list, norm_data),
            "zentralcodes": (zentral_codes_list, zentral_codes_data),
            "schlussel": (schlussel_list, schlussel_data),
            "zentral_middle": zentral_middle,
            "normal_middle": normal_middle,
            "code": code,
            "date": date,
            "lookup_table": lookup_table,
            "schlussel_indices":schlussel_indices
        }
        # print(final_data["schlussel_indices"])
        for index, value in enumerate(final_data["zentralzylinder"][0]):
            if value in lookup_table.keys():
                final_data["zentral_middle"][index] = lookup_table[value]
                
        #print(final_data["schlussel"][1][0])
        #print(final_data["schlussel"][0])
        #print(final_data["zentralcodes"][1][3])
        #print(final_data["normalzylinder"][0])
        #print(final_data["zentralzylinder"][1][4])
        #print(final_data["lookup_table"])
        #print(final_data["schlussel_indices"])
        return final_data

if __name__ == '__main__':
    print("read_cols.py invoked")
    #read_both()
