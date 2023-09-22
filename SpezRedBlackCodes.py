import numpy as np 
from itertools import combinations

def print_bin_str(arr,length):
    print()
    for d in arr:
        print(np.binary_repr(d, width=length))
    print()
    print("size of input arr:", len(arr))
    print()
D = [7, 11, 19, 35, 67, 131, 13, 21, 37, 69, 133, 25, 41, 73, 137, 49, 81, 145, 97, 161, 193, 14, 22, 38,
              70, 134, 26, 42, 74, 138, 50, 82, 146, 98, 162, 194, 28, 44, 76, 140, 52, 84, 148, 100, 164, 196, 56,
              88, 152, 104, 168, 200, 112, 176, 208, 224]
              
def insert_zero_bit(D, bit_pos_7):
    if bit_pos_7 == 0:
        D_7 = [d << 1 for d in D]
    elif bit_pos_7 == 8:
        D_7 = D
    else:
        left = 2 ** 8 - 2 ** bit_pos_7
        righ = 2 ** bit_pos_7 - 1
        D_7 = [((d & left) << 1) + (d & righ) for d in D]
    return D_7


def generate_numbers(n_unset, fix, has_fix=True):

    bits = [p for p in range(9) if p != fix]
    
    D_partial = []
    
    for bit_positions in combinations(bits, n_unset):
        number = 0b111111111
        for bit_position in bit_positions:
            number &= ~(1 << bit_position)
        D_partial.append(number)
        
    if has_fix:
        D_partial = [number & ~(1 << fix) for number in D_partial]
        
    return D_partial
    
def create_all_D_part_pairs(bit_pos_7_red, bit_pos_7_black):
    all_D_part_pairs = []
    
    for n_unset_red in range(1, 5):
        n_unset_bla = 5 - n_unset_red
        
        D_part_red = generate_numbers(n_unset_red, bit_pos_7_red, has_fix=True)
        D_part_bla = generate_numbers(n_unset_bla, bit_pos_7_black, has_fix=False)
        
        all_D_part_pairs.append((D_part_red, D_part_bla, n_unset_red, n_unset_bla, 'red'))
        
        D_part_red = generate_numbers(n_unset_red, bit_pos_7_red, has_fix=False)
        D_part_bla = generate_numbers(n_unset_bla, bit_pos_7_black, has_fix=True)
        
        all_D_part_pairs.append((D_part_red, D_part_bla, n_unset_red, n_unset_bla, 'bla'))
    
    return all_D_part_pairs


def get_mixed_D(D_red, D_bla):
    mask_red1 = 0b111100000
    mask_red2 = 0b000011111
    mask_bla1 = 0b111110000
    mask_bla2 = 0b000001111
    
    result = []
    
    for d_red in D_red:
        for d_bla in D_bla:

            d_red1 = d_red & mask_red1
            d_red2 = d_red & mask_red2
        
            d_bla1 = d_bla & mask_bla1
            d_bla2 = d_bla & mask_bla2
        
            if all((d_red1 & (1 << k)) or ((d_bla1 & (1 << k)) and (d_bla1 & (1 << k-1))) for k in range(8, 4, -1)) and all((d_bla2 & (1 << k)) or ((d_red2 & (1 << k)) and (d_red2 & (1 << k+1))) for k in range(4)):
                result.append((d_red, d_bla))
    
    return result
    

def compute_all_D_mixed(all_D_part_pairs):
    all_D_mixed = []
    
    for D_part_red, D_part_bla, _, _, _ in all_D_part_pairs:
        D_mixed = get_mixed_D(D_part_red, D_part_bla)
        all_D_mixed.append(D_mixed)
    
    return all_D_mixed
    
def get_all_red_black_spez_codes(bit_pos_7_red, bit_pos_7_black):
    all_D_part_pairs = create_all_D_part_pairs(bit_pos_7_red, bit_pos_7_black)
    all_D_mixed = compute_all_D_mixed(all_D_part_pairs)
    all_D_mixed_codes = []
    for li in all_D_mixed:
        codes = []
        for tup in li:
            a, b = tup 
            all_D_mixed_codes.append((a << 9) + b)
    return all_D_mixed_codes
    
#sample execution
#bit_pos_7_red, bit_pos_7_black = 4, 5
#all_D_mixed_codes = get_all_red_black_spez_codes(bit_pos_7_red, bit_pos_7_black)
#print_bin_str(all_D_mixed_codes, 18)

if __name__ == '__main__':
    print("SpezRedBlackCodes.py invoked")
    
    
    
    

