//a simple OpenCL kernel which adds two vectors A and B together into a third vector C
__kernel void add(__global const int* A, __global const int* B, __global int* C) {
	int id = get_global_id(0);
	C[id] = A[id] + B[id];
}

__kernel void mult(__global const int* A, __global const int* B, __global int* C) {
	int id = get_global_id(0);
	C[id] = A[id] * B[id];
}

__kernel void multF(__global const float* A, __global const float* B, __global float* C) {
	int id = get_global_id(0);
	C[id] = A[id] * B[id];
}

__kernel void addF(__global const float* A, __global const float* B, __global float* C) {
	int id = get_global_id(0);
	C[id] = A[id] + B[id];
}