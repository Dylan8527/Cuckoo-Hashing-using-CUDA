'''
Author: Dylan8527 vvm8933@gmail.com
Date: 2022-12-12 16:54:45
LastEditors: Dylan8527 vvm8933@gmail.com
LastEditTime: 2022-12-13 23:41:09
FilePath: \CS121_Fall2022_Lab2_chenqh_2020533088\tests\lookup\gen_test_file.py
Description: 

Copyright (c) 2022 by Dylan8527 vvm8933@gmail.com, All Rights Reserved. 
'''
for i in range(11):
    for n_h in range(2, 4):
        header = '''
#include "test.cuh"
'''
        func = f'std::string do_test2_{i}_{n_h}() ' + '{' + f' return do_test2<{i}, {n_h}>(); ' + '}'
        with open(f'test_{i}_{n_h}.cu', 'w') as f:
            print(header + func, file=f)
