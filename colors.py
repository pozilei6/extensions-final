# Import the necessary packages
import pdfplumber
import re
import os

# reduces row to required elements
def fetch_row(row):
    res = []
    for index, element in enumerate(row[:18]):
        if index in [3, 7, 11, 15]:
            continue
        else:
            res.append(element)
    return res


# parses h
def parse_H(row):
    black_H = [1, 1, 1, 1, 1]
    green_H = [1, 1, 1, 1, 1]
    red_H = [1, 1, 1, 1]
    for index, value in enumerate(row):
        if value != "":
            if index % 3 == 0:
                black_H[index // 3] = 0
            elif index % 3 == 1:
                green_H[index // 3] = 0
            elif index % 3 == 2:
                red_H[index // 3] = 0
    return [black_H, green_H, red_H]


# parses s1
def parse_S1(row):
    black_S1 = [1, 1, 1, 1, 1]
    green_S1 = [1, 1, 1, 1, 1]
    red_S1 = [1, 1, 1, 1]
    for index, value in enumerate(row):
        if value != "":
            if index % 3 == 0:
                black_S1[index // 3] = 0
            elif index % 3 == 1:
                green_S1[index // 3] = 0
            elif index % 3 == 2:
                red_S1[index // 3] = 0
    return [black_S1, green_S1, red_S1]


# parses s2
def parse_S2(row):
    black_S2 = [1, 1, 1, 1]
    green_S2 = [1, 1, 1, 1, 1]
    red_S2 = [1, 1, 1, 1, 1]
    for index, value in enumerate(row):
        if value != "":
            if index % 3 == 0:
                red_S2[index // 3] = 0
            elif index % 3 == 1:
                green_S2[index // 3] = 0
            elif index % 3 == 2:
                black_S2[index // 3] = 0
    return [red_S2, green_S2, black_S2]


# helper functino for parsing row
def parse_row(row, index):
    if index % 3 == 0:
        return parse_H(row)
    elif index % 3 == 1:
        return parse_S1(row)
    else:
        return parse_S2(row)
    pass


# returns a dictionary of 4 keys, each having results of that pattern
def read_colors():

    data_location = "Data/"
    files_location = []

    for file in os.listdir(data_location):
        if file.endswith(".pdf") and file != "ny4.pdf":
            files_location.append(os.path.join(data_location, file))

    for file in files_location:
        print(file)
        with pdfplumber.open(file) as pdf:
            pages = pdf.pages

            final_result = {
                "schl_codes": [],
                "zentrco_codes": [],
                "zentralzylinder": [],
                "normalzylinder": [],
            }

            for page_ind, pdf_page in enumerate(pages):
                print(f"Reading page {page_ind}")

                # extracts single page text
                single_page_text = pdf_page.extract_text()

                # sets key of dict
                if "Referenzliste Zentralzylinder-Zentralcodes" in single_page_text:
                    continue
                elif "Zentralcodes" in single_page_text:
                    mode = "zentrco_codes"
                elif "Schlüssel" in single_page_text:
                    mode = "schl_codes"
                elif "Zentralzylinder" in single_page_text:
                    mode = "zentralzylinder"
                elif "Normalzylinder" in single_page_text:
                    mode = "normalzylinder"
                else:
                    mode = None

                if mode is not None:
                    table = pdf_page.extract_table()
                    flag = True
                    for index, row in enumerate(table):
                        if index % 3 == 0 and row[0] == "":
                            break

                        reduced = fetch_row(row[5:])
                        result = parse_row(reduced, index)

                        # adds result to dict, creates a new entry if it is H row
                        if index % 3 == 0:
                            final_result[mode].append(result)
                        elif index % 3 == 1 or index % 3 == 2:
                            final_result[mode][-1].extend(result)

        # print("Printing length of Zentralcodes:\t", len(final_result["zentrco_codes"]))
        # print("Printing length of Schlüssel:\t", len(final_result["schl_codes"]))
        # print(
        #     "Printing length of Normalzylinder:\t", len(final_result["normalzylinder"])
        # )
        # print(
        #     "Printing length of Zentralzylinder", len(final_result["zentralzylinder"])
        # )
        # 
        # print(
        #     "The format for each value in the above lists is [black_H, green_H, red_H, black_S1, green_S1, red_S1, red_S2, green_S2, black_S2]"
        # )
        #
        # print("Example: printing first value of Zentralzylinder:")
        # print(final_result["normalzylinder"][0])
        # print(final_result["normalzylinder"][20])
        # print(final_result["normalzylinder"][21])
        # print(final_result["normalzylinder"][-1])
        #
        # print("\nprinting sample schl codes given in pdf:")
        # print(final_result["schl_codes"][2])
        # print(final_result["schl_codes"][1])

        return final_result
    
    
# appending colors of extension rows
def append_colors(colors_data, final_data, mode, final_m):
    
    m = len(colors_data[mode])
    for table in final_data[mode][1][m:]:
        
        hk, s1, s2 = table
        
        black_H =  [0 if hk[k] is not None and hk[k] != '' else 1 for k in range(0, 17, 4)] #[k for k in range(17) if k % 4 == 0]]
        green_H =  [0 if hk[k] is not None and hk[k] != '' else 1 for k in range(1, 18, 4)] #[k for k in range(18) if k % 4 == 1]]
        red_H   =  [0 if hk[k] is not None and hk[k] != '' else 1 for k in range(2, 15, 4)] #[k for k in range(15) if k % 4 == 2]]
        
        black_S1 = [0 if s1[k] is not None and s1[k] != '' else 1 for k in range(0, 17, 4)] #[k for k in range(17) if k % 4 == 0]]
        green_S1 = [0 if s1[k] is not None and s1[k] != '' else 1 for k in range(1, 18, 4)] #[k for k in range(18) if k % 4 == 1]]
        red_S1 =   [0 if s1[k] is not None and s1[k] != '' else 1 for k in range(2, 15, 4)] #[k for k in range(15) if k % 4 == 2]]
        
        black_S2 = [0 if s2[k] is not None and s2[k] != '' else 1 for k in range(2, 15, 4)] #[k for k in range(15) if k % 4 == 2]]
        green_S2 = [1, 1, 1, 1, 1]
        red_S2 =   [1, 1, 1, 1, 1]
        
        colors_data[mode].append([black_H, green_H, red_H, black_S1, green_S1, red_S1, black_S2, green_S2, red_S2])
        
        """
        col_dat = []
        
        for k, value in enumerate(hk[:15]):
            if value != "" and value is not None:
                

        for index, row in enumerate(table):    
            reduced = fetch_row(row[5:]) 
            print(row)
            print(reduced)
            result = parse_row(reduced, index)
            col_dat += result
        colors_data[mode].append(col_dat)
        print()
        """



# read_colors()








