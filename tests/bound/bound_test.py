'''
Author: Dylan8527 vvm8933@gmail.com
Date: 2022-12-12 16:54:45
LastEditors: Dylan8527 vvm8933@gmail.com
LastEditTime: 2022-12-13 23:38:01
FilePath: \CS121_Fall2022_Lab2_chenqh_2020533088\tests\bound_test\bound_test.py
Description: 

Copyright (c) 2022 by Dylan8527 vvm8933@gmail.com, All Rights Reserved. 
'''
def gen(i, n_h):
    header = \
'''
#include "test.cuh"
'''
    func = f'std::string do_test4_{i}_{n_h}() ' + '{' + \
           f' return do_test4<{i}, {n_h}>(); ' \
           + '}'
    with open(f'test_{i}_{n_h}.cu', 'w') as f:
        print(header + func, file=f)

for n_h in range(2, 4):
    gen(2, n_h)
    gen(4, n_h)
    gen(6, n_h)
    gen(8, n_h)
    gen(10, n_h)
    gen(12, n_h)
    gen(14, n_h)
    gen(16, n_h)
    gen(18, n_h)
    gen(20, n_h)
    gen(22, n_h)
    gen(24, n_h)
    gen(24, n_h)
    gen(36, n_h)
    gen(48, n_h)
    gen(72, n_h)
    gen(96, n_h)
    gen(144, n_h)