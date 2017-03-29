#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "Utils.h"
#include <chrono>

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

float* read(const char* dir, int size)
{
	float* tmp = new float[size];

	FILE* stream = fopen(dir, "r");
	fseek(stream, 0L, SEEK_SET);
	
	for (int i = 0; i < size; i++)
		fscanf(stream, "%*s %*lf %*lf %*lf %*lf %f", &tmp[i]);

	fclose(stream);

	return tmp;
}

void PaddingFloat(std::vector<float> &A, int local_size)
{
	size_t padding_size = A.size() % local_size;
	if (padding_size) {
		std::vector<int> A_ext(local_size - padding_size, 0);
		A.insert(A.end(), A_ext.begin(), A_ext.end());
	}
}


int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++)	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

	//detect any potential exceptions
	try {
		cl::Context context = GetContext(platform_id, device_id);
		std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		cl::Program::Sources sources;
		AddSources(sources, "my_kernels3.cl");
		cl::Program program(context, sources);

		std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
		//Read files
		int const size = 1873106;
		//1873106
		cout << "--------------------------------------------------------------------------------" << endl;
		float* Temprature = read("../temp_lincolnshire.txt", size);
		cout << "\tRead and Parse[ms]: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now).count() << endl;
		cout << "--------------------------------------------------------------------------------" << endl;

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}
		typedef float mytype;

		//create vector of floats from data
		std::vector<mytype> A(size);
		for (int i = 0;	 i < size; i++){A[i] = Temprature[i];}

		/* Make the data set 5.2 BILLION
		A.insert(A.end(), A.begin(), A.end());A.insert(A.end(), A.begin(), A.end());A.insert(A.end(), A.begin(), A.end());A.insert(A.end(), A.begin(), A.end());A.insert(A.end(), A.begin(), A.end());A.insert(A.end(), A.begin(), A.end());A.insert(A.end(), A.begin(), A.end());A.insert(A.end(), A.begin(), A.end());
		*/

		//the following part adjusts the length of the input vector so it can be run for a specific workgroup size
		//if the total input length is divisible by the workgroup size
		//this makes the code more efficient
		size_t local_size = 128;

		PaddingFloat(A,local_size);

		size_t input_elements = A.size();//number of input elements
		size_t input_size = A.size() * sizeof(mytype);//size in bytes
		size_t nr_groups = input_elements / local_size;	

		//host - output
		std::vector<mytype> Mean(nr_groups);
		size_t output_size = Mean.size() * sizeof(mytype);//size in bytes

		//host - output
		std::vector<mytype> Max(nr_groups);
		std::vector<mytype> Min(nr_groups);
		std::vector<mytype> Var(input_size);

		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_Max(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_Min(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_Var(context, CL_MEM_READ_WRITE, input_size);
		
		cl::Event prof_event;

		//Part 5 - device operations
		//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		//5.1 copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory

		//5.2 Setup and execute all kernels (i.e. device code)
		cl::Kernel kernel_1 = cl::Kernel(program, "total_Add");
		kernel_1.setArg(0, buffer_A);
		kernel_1.setArg(1, buffer_B);
		kernel_1.setArg(2, cl::Local(local_size*sizeof(mytype)));//local memory size

		//call all kernels in a sequence
		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size),NULL, &prof_event);

		//5.3 Copy the result from device to host
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &Mean[0]);

		std::chrono::high_resolution_clock::time_point Addtime = std::chrono::high_resolution_clock::now();
		mytype total = 0; for (int i = 0; i <= nr_groups; i++) { total += Mean[i]; }
		cout << "\tHost Add Time[ns]: " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - Addtime).count() << endl;
		std::cout << "\tMean = " << total / size << std::endl;
		std::cout << "\t" << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << endl;
		std::cout << "\tKernel execution time[ns]:" << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		cout << "--------------------------------------------------------------------------------" << endl;
		
		
		//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueFillBuffer(buffer_Max, 0, 0, output_size);

		cl::Kernel kernel_Max = cl::Kernel(program, "Maximum_Local");
		kernel_Max.setArg(0, buffer_A);
		kernel_Max.setArg(1, buffer_Max);
		kernel_Max.setArg(2, cl::Local(local_size * sizeof(mytype)));

		queue.enqueueNDRangeKernel(kernel_Max, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);
		queue.enqueueReadBuffer(buffer_Max, CL_TRUE, 0, output_size, &Max[0]);

		std::chrono::high_resolution_clock::time_point Maxtime = std::chrono::high_resolution_clock::now();
		mytype MaxValue = 0; for (int i = 0; i <= nr_groups; i++) { if (MaxValue < Max[i]) MaxValue = Max[i]; }
		cout << "\tHost Time[ns]: " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - Maxtime).count() << endl;
		std::cout << "\tMax = " << MaxValue << std::endl;
		std::cout << "\t" << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << endl;
		std::cout << "\tKernel execution time[ns]:" << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		cout << "--------------------------------------------------------------------------------" << endl;

		//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueFillBuffer(buffer_Min, 0, 0, output_size);

		cl::Kernel kernel_Min = cl::Kernel(program, "Minimum_Local");
		kernel_Min.setArg(0, buffer_A);
		kernel_Min.setArg(1, buffer_Min);
		kernel_Min.setArg(2, cl::Local(local_size * sizeof(mytype)));

		queue.enqueueNDRangeKernel(kernel_Min, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);
		queue.enqueueReadBuffer(buffer_Min, CL_TRUE, 0, output_size, &Min[0]);
	
		std::chrono::high_resolution_clock::time_point Mintime = std::chrono::high_resolution_clock::now();
		mytype MinValue = 0;for (int i = 0; i <= nr_groups; i++) { if (MinValue > Min[i]) MinValue = Min[i]; }
		cout << "\tHost Time[ns]: " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - Mintime).count() << endl;
		std::cout << "\tMin = " << MinValue << std::endl;
		std::cout << "\t" << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << endl;
		std::cout << "\tKernel execution time[ns]:" << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		cout << "--------------------------------------------------------------------------------" << endl;
				
		//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueFillBuffer(buffer_Var, 0, 0, output_size);

		cl::Kernel kernel_Var = cl::Kernel(program, "Variance");
		kernel_Var.setArg(0, buffer_A);
		kernel_Var.setArg(1, buffer_Var);
		kernel_Var.setArg(2, total/size);

		queue.enqueueNDRangeKernel(kernel_Var, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);
		queue.enqueueReadBuffer(buffer_Var, CL_TRUE, 0, input_size, &Var[0]);

		std::cout << "\tVariance = " << Var[0] << std::endl;
		std::cout << "\t" << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << endl;
		std::cout << "\tKernel execution time[ns]:" << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		cout << "--------------------------------------------------------------------------------" << endl;
		
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	system("Pause");
	return 0;
}


/*
	atomic floats:http://suhorukov.blogspot.co.uk/2011/12/opencl-11-atomic-operations-on-floating.html
	
	Optimal Work Group Size: 128

	Times (NS)
	
	Addition:	time[ns]:96640
	Mean:		time[ns]:17003
	Add + M (1070):time[ns]:66560

	Min Local:  time[ns]:145408
	Max Local:		
 	

	Min Global Int:	time[ns]:195808				
	Max Global Int:	time[ns]:1423040


	--Tested on 1.2B--
	Unwrapped:  time[ns]:5432928
	Atomic:		time[ns]:4989824

	--Tested on 5.2B--
	Unwrapped:  time[ns]:21715776
	Atomic:	    time[ns]:20010144


*/
