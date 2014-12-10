export PYOPENCL_CTX="1"
export CPU_MAX_COMPUTE_UNITS=8
./CRPkdTimeTest.py --iter 10000 --dim 1 --output_to_file --cluster_num 2 --opencl
./CRPkdTimeTest.py --iter 10000 --dim 1 --output_to_file --cluster_num 3 --opencl
./CRPkdTimeTest.py --iter 10000 --dim 1 --output_to_file --cluster_num 5 --opencl
./CRPkdTimeTest.py --iter 10000 --dim 1 --output_to_file --cluster_num 10 --opencl
./CRPkdTimeTest.py --iter 10000 --dim 1 --output_to_file --cluster_num 25 --opencl
./CRPkdTimeTest.py --iter 10000 --dim 1 --output_to_file --cluster_num 100 --opencl

./CRPkdTimeTest.py --iter 10000 --dim 2 --output_to_file --cluster_num 2 --opencl
./CRPkdTimeTest.py --iter 10000 --dim 2 --output_to_file --cluster_num 3 --opencl
./CRPkdTimeTest.py --iter 10000 --dim 2 --output_to_file --cluster_num 5 --opencl
./CRPkdTimeTest.py --iter 10000 --dim 2 --output_to_file --cluster_num 10 --opencl
./CRPkdTimeTest.py --iter 10000 --dim 2 --output_to_file --cluster_num 25 --opencl
./CRPkdTimeTest.py --iter 10000 --dim 2 --output_to_file --cluster_num 100 --opencl

./CRPkdTimeTest.py --iter 10000 --dim 3 --output_to_file --cluster_num 2 --opencl
./CRPkdTimeTest.py --iter 10000 --dim 3 --output_to_file --cluster_num 3 --opencl
./CRPkdTimeTest.py --iter 10000 --dim 3 --output_to_file --cluster_num 5 --opencl
./CRPkdTimeTest.py --iter 10000 --dim 3 --output_to_file --cluster_num 10 --opencl
./CRPkdTimeTest.py --iter 10000 --dim 3 --output_to_file --cluster_num 25 --opencl
./CRPkdTimeTest.py --iter 10000 --dim 3 --output_to_file --cluster_num 100 --opencl
