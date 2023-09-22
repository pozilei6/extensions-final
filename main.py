import tkinter as tk
import extension as ex



def get_substring(str_Inz):
    lines = str_Inz.split('\n')
    non_empty_lines = [line for line in lines if line.strip() != '']
    return '\n'.join(non_empty_lines[1:])

#---------------
"""
def get_new_zyl_from_user_input(n: int, str_Inz_ex: str) -> list: # for new normalzylinder from user input, n=len(A[0])

    # e.g. str_Inz_ex = 'A4 B4 5   \nA4 5\n   6   C45    dfe  B4 \n 8   7', n=44
    #
    #      returns Inz_ex = [{5, 45, 47}, {5, 45}, {46, 48, 6, 47}, {8, 7}] 
    
    substrings = str_Inz_ex.split('\n')
    substrings = [subst.split(' ') for subst in substrings]
    substrings = [[su for su in subst if su != ''] for subst in substrings]
    substrings = [subst for subst in substrings if subst]
    
    all_substr = set().union(*substrings)
    newC = [s for s in all_substr if not s.isdigit()]
    new_C_range = range(n + 1, n + 1 + len(newC))
    
    exiC_dict = {s: int(s) for s in all_substr if s.isdigit()}
    newC_dict = {new_str: new_int for (new_str, new_int) in zip(newC, new_C_range)}
    
    exiC_dict.update(newC_dict)
    exi_new_dict = exiC_dict
    
    Inz_ex = [{exi_new_dict[su] for su in subst} for subst in substrings]

    return Inz_ex, newC_dict# <- used in get_extended_existing_zyl_from_user_input()
    
    
def get_extended_existing_zyl_from_user_input(str_InzC_ex, newC_dict, n):
    # zyl 1 additionally opened by new A4, A5 and existing 5, zyl 2 additionally opened by existing 10
    # e.g. str_InzC_ex = '1: A4 A5 5   \n  2:  10'
    #      newC_dict = {'C45': 45, 'dfe': 46, 'B4': 47, 'A4': 48}   from get_new_zyl_from_user_input()[1]
    #      n = 44
    #      returns Inz_ex = [1: {46, 49, 5}, 2: {10}] 
    
    substrings = str_InzC_ex.split('\n')
    substrings = [su for su in substrings if su.strip()]
    substrings = [subst.split(' ') for subst in substrings]
    substrings = [[su for su in subst if su != ''] for subst in substrings]
    
    zyl_numbers = [int(sub[0].strip(':')) for sub in substrings]
    
    zyl_schluss = [[su for su in subst if ':' not in su] for subst in substrings]
    
    all_substr = set().union(*zyl_schluss)
    int_substr = [s for s in all_substr if s.isdigit()]
    nin_substr = [s for s in all_substr if s not in int_substr]
    int_dict = {s: int(s) for s in int_substr}
    
    # appending new schlüssel-numbers to dict starting from max + 1 of newC_dict.values
    next_schl_numb = max(newC_dict.values()) + 1 if newC_dict.values() else n + 1 # last new schlüssel-number already assigned in newC_dict + 1
    InzC_ex_dict = newC_dict
    for s in nin_substr:
        if s not in InzC_ex_dict.keys():
            InzC_ex_dict[s] = next_schl_numb
            next_schl_numb += 1
            
    InzC_ex_dict.update(int_dict) # here we have all new schl-strings that existed in 
    
    InzC_ex = {z_num: {InzC_ex_dict[su] for su in zyl_schl} for z_num, zyl_schl in zip(zyl_numbers, zyl_schluss)}
    
    return InzC_ex


def get_Inz_ex_InzC_ex(n, str_Inz_ex, str_InzC_ex): 
    Inz_ex, newC_dict = get_new_zyl_from_user_input(n, str_Inz_ex)
    InzC_ex = get_extended_existing_zyl_from_user_input(str_InzC_ex, newC_dict, n)
    return Inz_ex, InzC_ex
""" 
#------------------

# initial values of variables
Inz_ex = []
InzC_ex = dict()
str_Inz_ex = ''
str_InzC_ex = ''


"""
# defining events
def set_extension_data():
    global var_str_Inz_ex
    global var_str_InzC_ex
    global txt_str_Inz_ex
    global txt_str_InzC_ex
    global str_Inz_ex
    global str_InzC_ex 
    global Inz_ex
    global InzC_ex
    global btn_solve_extension
    n = 57 #need to be read later    get_substring(str_Inz)
    if var_bool_ext_data.get():
        str_Inz_ex = get_substring(txt_str_Inz_ex.get('1.0','end')) #var_str_Inz_ex.get()
        str_InzC_ex = get_substring(txt_str_InzC_ex.get('1.0','end')) #var_str_InzC_ex.get()
        print(str_Inz_ex)
        print(str_InzC_ex)
        #Inz_ex, InzC_ex = get_Inz_ex_InzC_ex(n, var_str_Inz_ex.get(), var_str_InzC_ex.get())
        btn_solve_extension["state"] = "normal"
    else:
        btn_solve_extension["state"] = "disabled"
"""
        
def solve_extension():
    global str_Inz_ex
    global str_InzC_ex
    
    global txt_str_all_inz
    global str_all_inz
    
    str_all_inz = get_substring(txt_str_all_inz.get('1.0','end'))
    
    global lbl_vorgehen
    global lbl_neufertigen
    
    global var_mixed
    
    lbl_vorgehen["text"] = ''
    lbl_neufertigen["text"] = ''
    #ex.solve_extension(str_Inz_ex, str_InzC_ex, lbl_vorgehen)
    ex.solve_extension(str_all_inz, lbl_vorgehen, lbl_neufertigen, var_mixed.get())
    
    #lbl_vorgehen["text"] = "Running ex.solve_extension(str_Inz_ex, str_InzC_ex) function..."
    

# Tkinter GUI code
window = tk.Tk()
window.title("Extensions")
window.geometry('900x400')



var_str_Inz_ex = tk.StringVar()
var_str_InzC_ex = tk.StringVar()
var_bool_ext_data = tk.BooleanVar()
var_mixed = tk.BooleanVar()



# widgets
"""
chb_set_extension_data = tk.Checkbutton(master=window, text="set extension data", variable=var_bool_ext_data, onvalue=True, offvalue=False, command=set_extension_data)
chb_set_extension_data.grid(row=6, column=2, sticky=tk.W, padx=40)
"""

btn_solve_extension = tk.Button(master=window, text="Solve Extension", command=solve_extension)
btn_solve_extension.grid(row=1, column=2, sticky=tk.NW, padx=0, pady=20)
#btn_solve_extension["state"] = "normal"


chb_mixed_red_black_codes = tk.Checkbutton(master=window, text="mixed codes", variable=var_mixed, onvalue=True, offvalue=False)  
var_mixed.set(False)
chb_mixed_red_black_codes.grid(row=5, column=1, sticky=tk.NW, padx=40, pady=0)


# for both textbox ... will replace other two
txt_str_all_inz = tk.Text(master=window, height = 10, width = 22)                                                                                                                                   # 09.07. 16:19
txt_str_all_inz.grid(row=1, column=1,  rowspan=4, sticky=tk.W, padx=40, pady=20)               # row=9, column=3
txt_str_all_inz.insert('1.0', 'N-Extension-Zylinders: \n')


lbl_vorgehen = tk.Label(master=window, text="", fg="blue", justify="left", font=('Segoe UI Variable', 10))
lbl_vorgehen.grid(row=3, column=2, columnspan=2, sticky=tk.W, padx=0)

lbl_neufertigen = tk.Label(master=window, text="", fg="blue", justify="left", font=('Segoe UI Variable', 10))
lbl_neufertigen.grid(row=2, column=2, columnspan=2, sticky=tk.NW)

window.mainloop()






















