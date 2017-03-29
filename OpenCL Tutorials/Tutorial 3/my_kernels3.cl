//Calculate Total 2 stage reduction
__kernel void total_Add(__global const float* A, __global float* B, __local float* scratch) {

	///Refrence: http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
	//Block size must always be of power of 2! and <= 128
	int id = get_global_id(0);
	//Data size
	int size = get_global_size(0);
	int lid = get_local_id(0);
	//get workgroup size
	int N = get_local_size(0);
	//get group index postion
	int Gid = get_group_id(0);

	//Halve the number of blocks and replace single load
	int I = Gid * (N*2) + lid;

	//Gridsize to control loop to maintain coalescing
	int gridSize = N*2*get_num_groups(0);
	 
	scratch[lid] = 0;

	//Mantains coalescing by keeping values close together in scratch using gridsize
	//Fist Sequential reduction during read into local scratch to save time
	while (I < size) {scratch[lid] = (A[I] + A[I+N]); I += gridSize;}

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory
	 
	//cascading algorithm that does parrallel reduction on remaning workgroup items based on the workgroup size
	//unrolled all the previous loops!

	//Checks based on Workgroup size and local Id 
	if (N >= 128) { if (lid <64) {scratch[lid] += scratch[lid + 64];}barrier(CLK_LOCAL_MEM_FENCE);} 

	//This saves work on useless values and only executes if it needs to
	if (lid < 32){
	if (N >= 64) scratch[lid] += scratch[lid+32];
	if (N >= 32) scratch[lid] += scratch[lid+16];
	if (N >= 16) scratch[lid] += scratch[lid+8];
	if (N >= 8) scratch[lid] += scratch[lid+4];
	if (N >= 4) scratch[lid] += scratch[lid+2];
	if (N >= 2) scratch[lid] += scratch[lid+1];
	}
	
	//copy the cache to output array for every workgroup total value
	if (lid == 0) {B[Gid] = scratch[0];}
}

/*
	//Interleaved addressing
	for (int i = 1; i < N; i *= 2) 
	{
		int index = 2 * i * lid;
		if(index < N) 
		{
			if (scratch[lid] < scratch[lid + i])
				scratch[lid] = scratch[lid+i];	
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//Sequential Addressing
	for (int i = N/2; i > 0; i >>= 1) 
	{
		if(lid < i) 
		{
			if (scratch[lid] < scratch[lid + i])
				scratch[lid] = scratch[lid+i];	
		}
		//time[ns]:221856
		barrier(CLK_LOCAL_MEM_FENCE);
	}
*/

__kernel void Variance(__global const float* A, __global float* B, float mean) {

	//Each Number subtracr the mean and square the result
	//Take squared diffrence and run total add
	int size = get_global_size(0);
	int id = get_global_id(0);

	if (id < size)
		B[id] = A[id] - mean;

	B[id] = B[id] * B[id];
}
///Atomic Functions-------------------------------------------------------------------
//Max Local
__kernel void Maximum_Local(__global float* A, __global float* B, __local float* scratch) 
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	//get workgroup size
	int N = get_local_size(0);
	//get group index postion
	int Gid = get_group_id(0);
	int size = get_global_size(0);

	//Halve the number of blocks and replace single load
	int I = Gid * (N*2) + lid;

	//Gridsize to control loop to maintain coalescing
	int gridSize = N*2*get_num_groups(0);
	 
	scratch[lid] = 0;

	//Mantains coalescing by keeping values close together in scratch using gridsize
	//Fist Sequential reduction during read into local scratch to save time
	while (I < size) { scratch[lid] = A[I]; if(scratch[lid] < A[I+N]) scratch[lid] = A[I+N];	I += gridSize;	}

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory
	 
	//cascading algorithm that does parrallel reduction on remaning workgroup items based on the workgroup size
	//unrolled all the previous loops!

	//Checks based on Workgroup size and local Id 
	if (N >= 128) { if (lid <64) {if(scratch[lid] < scratch[lid+64]) scratch[lid] = scratch[lid+64];}barrier(CLK_LOCAL_MEM_FENCE);} 

	if (lid < 32){
		if (N >= 64) if(scratch[lid] < scratch[lid+32]) scratch[lid] = scratch[lid+32];
		if (N >= 32) if(scratch[lid] < scratch[lid+16]) scratch[lid] = scratch[lid+16];
		if (N >= 16) if(scratch[lid] < scratch[lid+8]) scratch[lid] = scratch[lid+8];
		if (N >= 8) if(scratch[lid] < scratch[lid+4]) scratch[lid] = scratch[lid+4];
		if (N >= 4) if(scratch[lid] < scratch[lid+2]) scratch[lid] = scratch[lid+2];
		if (N >= 2) if(scratch[lid] < scratch[lid+1]) scratch[lid] = scratch[lid+1];
	}
	
	if (lid == 0) {B[Gid] = scratch[lid];barrier(CLK_LOCAL_MEM_FENCE);}

	/*
	//Quicker to not do this with ints
	for (int i = 1; i < N; i *= 2) 
	{
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			if(scratch[lid+i] > scratch[lid])	
				scratch[lid] = scratch[lid+i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	atomic_max(&B[0], scratch[lid]*10);
	*/
}

//REDUCE METHOD
__kernel void Minimum_Local(__global int* A, __global int* B, __local int* scratch) 
{
	int lid = get_local_id(0);	int N = get_local_size(0);
	int Gid = get_group_id(0);	int size = get_global_size(0);

	int I = Gid * (N*2) + lid;

	int gridSize = N*2*get_num_groups(0);
	 
	scratch[lid] = 0;

	while (I < size) { scratch[lid] = A[I]; if(scratch[lid] > A[I+N]) scratch[lid] = A[I+N];	I += gridSize;	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (N >= 128) { if (lid <64) {if(scratch[lid] > scratch[lid+64]) scratch[lid] = scratch[lid+64];}barrier(CLK_LOCAL_MEM_FENCE);} 

	if (lid < 32){
		if (N >= 64) if(scratch[lid] > scratch[lid+32]) scratch[lid] = scratch[lid+32];
		if (N >= 32) if(scratch[lid] > scratch[lid+16]) scratch[lid] = scratch[lid+16];
		if (N >= 16) if(scratch[lid] > scratch[lid+8]) scratch[lid] = scratch[lid+8];
		if (N >= 8) if(scratch[lid] > scratch[lid+4]) scratch[lid] = scratch[lid+4];
		if (N >= 4) if(scratch[lid] > scratch[lid+2]) scratch[lid] = scratch[lid+2];
		if (N >= 2) if(scratch[lid] > scratch[lid+1]) scratch[lid] = scratch[lid+1];
	}
	
	if (lid == 0) {B[Gid] = scratch[lid];barrier(CLK_LOCAL_MEM_FENCE);}
}
//Max_Global
__kernel void Maximum_Global_Int(__global int* A, __global int* B) 
{
	int id = get_global_id(0); 
	int N = get_local_size(0);

	B[id] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	atomic_max(&B[0], A[id]);
}
//REDUCE METHOD
__kernel void Minimum_Global(__global int* A, __global int* B) 
{
	int id = get_global_id(0); 
	int N = get_local_size(0);

	B[id] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory
	
	atomic_min(&B[0], A[id]);
}

//REDUCE METHOD
__kernel void Atomic_Add(__global int* A, __global int* B, __local int* scratch) 
{
	int id = get_global_id(0); 
	int N = get_local_size(0);
	int lid = get_local_id(0);

	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory
	
	atomic_add(&B[0], scratch[lid]);
}


/*
//Tried and failed sorting
void cmpxchg(__global int* A, __global int* B, bool dir) 
{
	if ((!dir && *A > *B) || (dir && *A < *B)) 	{
		int t = *A;
		*A = *B;
		*B = t;
	}
}

///BITONIC SORT SORT---------------------------------------------------------------------
void bitonic_merge(int id, __global int* A, int N, bool dir) 
{
	for (int i = N/2; i > 0; i/=2)
	{
		if ((id % (i*2)) < i)
		{ 
			cmpxchg(&A[id],&A[id+i],dir);
		}

		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}

__kernel void ParallelSelection(__global const int* A)
{
  int id = get_global_id(0); // current thread
  int N = get_local_size(0); // input size

  for (int i = 1; i < N/2; i*=2)
  {
		if (id % (i*4) < i*2)
		{    
			bitonic_merge(id, A, i*2, false);
		}
		else if ((id + i*2) % (i*4) < i*2)
		{ 
		bitonic_merge(id, A, i*2, true);
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
	bitonic_merge(id,A,N,false);
}


///ODD EVEN SORT---------------------------------------------------------------------
void OddEvenSort(__global int* A, __global int* B)
{
	if (*A > *B) 
	{ 
		int t = *A; 
		*A = *B; 
		*B = t;
	}
}

__kernel void sort_oddeven(__global int* A, __global int* B) 
{
	int id = get_global_id(0); 
	int N = get_local_size(0);
	//int lid = get_local_id(0);

	//Scratch[lid] = A[id];

	for (int i = 0; i < N; i+=2) 
	{//step
		if (id%2 == 1 && id+1 < N) //odd
		{
			OddEvenSort(&A[id],&A[id+1]);
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
		if (id%2 == 0 && id+1 < N) //even
		{ 
			OddEvenSort(&A[id],&A[id+1]);
		}
	}
	B[id] = A[id];
}


//fixed 4 step reduce
__kernel void reduce_add_1(__global const int* A, __global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);

	B[id] = A[id]; //copy input to output

	barrier(CLK_GLOBAL_MEM_FENCE); //wait for all threads to finish copying
	 
	//perform reduce on the output array
	//modulo operator is used to skip a set of values (e.g. 2 in the next line)
	//we also check if the added element is within bounds (i.e. < N)
	if (((id % 2) == 0) && ((id + 1) < N)) 
		B[id] += B[id + 1];

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (((id % 4) == 0) && ((id + 2) < N)) 
		B[id] += B[id + 2];

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (((id % 8) == 0) && ((id + 4) < N)) 
		B[id] += B[id + 4];

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (((id % 16) == 0) && ((id + 8) < N)) 
		B[id] += B[id + 8];
}

//flexible step reduce 
__kernel void reduce_add_2(__global const int* A, __global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);

	B[id] = A[id];

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2) { //i is a stride
		if (!(id % (i * 2)) && ((id + i) < N)) 
			B[id] += B[id + i];

		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}

//reduce using local memory (so called privatisation)
__kernel void reduce_add_3(__global const int* A, __global int* B, __local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//copy the cache to output array
	B[id] = scratch[lid];
}

//reduce using local memory + accumulation of local sums into a single location
//works with any number of groups - not optimal!
__kernel void reduce_add_4(__global const int* A, __global int* B, __local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid) {
		atomic_add(&B[0],scratch[lid]);
	}
}

//a very simple histogram implementation
__kernel void hist_simple(__global const int* A, __global int* H) { 
	int id = get_global_id(0);

	//assumes that H has been initialised to 0
	int bin_index = A[id];//take value as a bin index

	atomic_inc(&H[bin_index]);//serial operation, not very efficient!
}

//a double-buffered version of the Hillis-Steele inclusive scan
//requires two additional input arguments which correspond to two local buffers
__kernel void scan_add(__global const int* A, __global int* B, __local int* scratch_1, __local int* scratch_2) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	__local int *scratch_3;//used for buffer swap

	//cache all N values from global memory to local memory
	scratch_1[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (lid >= i)
			scratch_2[lid] = scratch_1[lid] + scratch_1[lid - i];
		else
			scratch_2[lid] = scratch_1[lid];

		barrier(CLK_LOCAL_MEM_FENCE);

		//buffer swap
		scratch_3 = scratch_2;
		scratch_2 = scratch_1;
		scratch_1 = scratch_3;
	}

	//copy the cache to output array
	B[id] = scratch_1[lid];
}

//calculates the block sums
__kernel void block_sum(__global const int* A, __global int* B, int local_size) {
	int id = get_global_id(0);
	B[id] = A[(id+1)*local_size-1];
}

//simple exclusive serial scan based on atomic operations - sufficient for small number of elements
__kernel void scan_add_atomic(__global int* A, __global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	for (int i = id+1; i < N; i++)
		atomic_add(&B[i], A[id]);
}

//adjust the values stored in partial scans by adding block sums to corresponding blocks
__kernel void scan_add_adjust(__global int* A, __global const int* B) {
	int id = get_global_id(0);
	int gid = get_group_id(0);
	A[id] += B[gid];
}




*/



