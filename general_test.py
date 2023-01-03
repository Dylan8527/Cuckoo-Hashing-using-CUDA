'''
Author: Dylan8527 vvm8933@gmail.com
Date: 2022-12-12 16:54:45
LastEditors: Dylan8527 vvm8933@gmail.com
LastEditTime: 2022-12-13 23:27:43
FilePath: \CS121_Fall2022_Lab2_chenqh_2020533088\general_test.py
Description: 

Copyright (c) 2022 by Dylan8527 vvm8933@gmail.com, All Rights Reserved. 
'''

import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str, default='workspace',help='print your workspace name (do not print full path)')
    parser.add_argument('--log_insert', '--number2insert', type=int, default=24, help='log number of entries to insert')
    parser.add_argument('--log_size', '--hash_table_size', type=int, default=25, help='log base size of hash table')
    parser.add_argument('--scale', type=float, default=1.0, help='scale of hash table size')
    parser.add_argument('--bound', type=int, default=32, help='evict chain bound for inserting a key')

    parser.add_argument('--insert', type=int, default=1 << parser.parse_args().log_insert, help='number of entries to insert')
    parser.add_argument('--size', type=int, default=int(parser.parse_args().scale * (1 << parser.parse_args().log_size)), help='size of hash table')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(args)

    project_dir = os.path.join(os.getcwd(), 'tests', args.workspace)
    os.makedirs(project_dir, exist_ok=True)

    # 1. create CMakeLists.txt
    cmake_project = \
    f'project({args.workspace} LANGUAGES CUDA)' 
    cmake_lines = \
    '''
file(GLOB SRC_FILES "*.cu")
add_library(${PROJECT_NAME} STATIC ${SRC_FILES})
target_compile_features(${PROJECT_NAME} PRIVATE cuda_std_17)
target_link_libraries(${PROJECT_NAME} PRIVATE HashTable::HashTable OpenMP::OpenMP_CXX)
    '''

    with open(os.path.join(project_dir, 'CMakeLists.txt'), 'w') as f:
        print(cmake_project+cmake_lines, file=f)

    # update CMakeLists outside in the './tests'
    # first check whether the workspace is already added
    flag = False
    with open(os.path.join(os.getcwd(), 'tests', 'CMakeLists.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            if args.workspace in line:
                flag = True
                break
    if not flag:
        with open(os.path.join(os.getcwd(), 'tests', 'CMakeLists.txt'), 'a') as f:
            print(f'add_subdirectory({args.workspace})', file=f)
            print(f'target_compile_features({args.workspace} PRIVATE cuda_std_17)')
            print(f'target_link_libraries({args.workspace} PRIVATE HashTable::HashTable OpenMP::OpenMP_CXX {args.workspace} correctness_test)')
 
