'''
Author: Dylan8527 vvm8933@gmail.com
Date: 2022-12-12 16:54:46
LastEditors: Dylan8527 vvm8933@gmail.com
LastEditTime: 2022-12-13 23:41:49
FilePath: \CS121_Fall2022_Lab2_chenqh_2020533088\tests\size\gen_test_file.py
Description: 

Copyright (c) 2022 by Dylan8527 vvm8933@gmail.com, All Rights Reserved. 
'''
def gen(i, n_h):
    header = '''
#include "test.cuh"
'''
    func = f'std::string do_test3_{i}_{n_h}() ' + '{' + \
           f' return do_test3<static_cast<std::uint32_t>((1 << 24) * {i/100:.2f} + 1 - 1e-10), {n_h}>(); ' \
           + '}'
    with open(f'test_{i}_{n_h}.cu', 'w') as f:
        print(header + func, file=f)

for i in range(110, 201, 10):
    for n_h in range(2, 4):
        gen(i, n_h)
gen(101, 2)
gen(102, 2)
gen(105, 2)
gen(101, 3)
gen(102, 3)
gen(105, 3)