# Cuckoo Hashing using CUDA

ShanghaiTech CS121 Parallel Computing Fall 2022 Lab2.  
High performance parallel hash table based on Cuckoo Hashing using CUDA.

​	A hash table is a data structure that is used to store keys and values, which allows for quick insertion and lookup of the values, making it a useful data structure for a variety of applications. Cuckoo hashing is a technique for implementing a hash table that uses a cuckoo search algorithm to efficiently store and retrieve key-value pairs by mapping keys to two different indices in an array and storing the values at those indices. In this lab, we implement Cuckoo Hashing by CUDA language, which is supported by several test experiments related to algorithm accuracy and performances.

## Build

```shell
mkdir build
cd build
cmake ..
cmake --build . --config Release -j
```

## Run

You can find the executable file in `./build/tests/Release` if you follow the above steps~

```shell
./main
```
or
```shell
.\main.exe
```


If you want to perform your own experiments, we provide a python script to create a workspace to run the program.
```shell
python generatl_test.py
```
## Experiments

See details in [report](report.pdf).
