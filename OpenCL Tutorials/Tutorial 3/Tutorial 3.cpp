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
int lines = 0;
float* read(const char* dir)
{
	FILE* stream = fopen(dir, "r");

	while (EOF != (fscanf(stream,"%*[^\n]"), fscanf(stream,"%*c")))
		++lines;

	float* tmp = new float[lines];

	fseek(stream, 0L, SEEK_SET);
	for (int i = 0; i < lines; i++)
	{
		fscanf(stream, "%*s %*lf %*lf %*lf %*lf %f", &tmp[i]);
	}

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
float total, MaxValue, MinValue, stdDev;

void ExecuteKernal(
	cl::CommandQueue queue,
	cl::Buffer IN,
	cl::Buffer Out,
	size_t input_size,
	std::vector<float> inVec,
	std::vector<float> OutVec,
	cl::Kernel kernel,
	int size,
	int local_size,
	size_t input_elements,
	cl::Event prof_event,
	int value
)
{
	int workgroups = (inVec.size() / local_size);

	queue.enqueueWriteBuffer(IN, CL_TRUE, 0, input_size, &inVec[0]);
	queue.enqueueFillBuffer(Out, 0, 0, OutVec.size());

	kernel.setArg(0, IN);
	kernel.setArg(1, Out);
	kernel.setArg(2, cl::Local(local_size * sizeof(float)));

	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(inVec.size()), cl::NDRange(local_size), NULL, &prof_event);
	
	queue.enqueueReadBuffer(Out, CL_TRUE, 0, OutVec.size() * sizeof(float) , &OutVec[0]);

	std::chrono::high_resolution_clock::time_point MeanTime, Maxtime, DevTime, Mintime;

	switch(value)
	{
	case 1:
		MeanTime = std::chrono::high_resolution_clock::now();
		total = 0; for (int i = 0; i < workgroups /2 ; i++) { total += OutVec[i]; }
		cout << "\tHost Time[ns]: " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - MeanTime).count() << endl;
		std::cout << "\tMean"<< " = " << total / size << std::endl;
		break;
	case 2:
		Maxtime = std::chrono::high_resolution_clock::now();
		MaxValue = 0; for (int i = 0; i < workgroups / 2; i++) { if (MaxValue < OutVec[i]) MaxValue = OutVec[i]; }
		cout << "\tHost Time[ns]: " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - Maxtime).count() << endl;
		std::cout << "\tMax = " << MaxValue << std::endl;
		break;
	case 3:
		Mintime = std::chrono::high_resolution_clock::now();
		MinValue = 0; for (int i = 0; i < workgroups / 2; i++) { if (MinValue > OutVec[i]) MinValue = OutVec[i]; }
		cout << "\tHost Time[ns]: " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - Mintime).count() << endl;
		std::cout << "\tMin = " << MinValue << std::endl;
		break;
	}
	std::cout << "\t" << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << endl;
	std::cout << "\tKernel execution time[ns]:" << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
	cout << "--------------------------------------------------------------------------------" << endl;
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
		
		//1873106
		cout << "--------------------------------------------------------------------------------" << endl;
		float* Temprature = read("../temp_lincolnshire.txt");
		cout << "\tRead and Parse[ms]: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now).count() << endl;
		cout << "--------------------------------------------------------------------------------" << endl;
		int const size = lines;

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

		// Make the data set 5.2 BILLION
		//A.insert(A.end(), A.begin(), A.end());A.insert(A.end(), A.begin(), A.end());A.insert(A.end(), A.begin(), A.end());A.insert(A.end(), A.begin(), A.end());A.insert(A.end(), A.begin(), A.end());A.insert(A.end(), A.begin(), A.end());A.insert(A.end(), A.begin(), A.end());A.insert(A.end(), A.begin(), A.end());
		

		//the following part adjusts the length of the input vector so it can be run for a specific workgroup size
		//if the total input length is divisible by the workgroup size
		//this makes the code more efficient
		size_t local_size = 128;
		PaddingFloat(A, local_size);
		size_t input_elements = A.size();//number of input elements
		size_t input_size = A.size() * sizeof(mytype);//size in bytes
		size_t nr_groups = input_elements / local_size;	

		//host - output
		std::vector<mytype> Mean(nr_groups);
		size_t output_size = Mean.size() * sizeof(mytype);//size in bytes

		//host - output
		std::vector<mytype> Max(nr_groups);
		std::vector<mytype> Dev(nr_groups);
		std::vector<mytype> Min(nr_groups);
		std::vector<mytype> Var(input_size);
		std::vector<mytype> Sort(input_size);

		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_Dev(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_Max(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_Min(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_Var(context, CL_MEM_READ_WRITE, input_size);
		cl::Buffer buffer_Sort(context, CL_MEM_READ_WRITE, input_size);
		
		cl::Event prof_event;

		//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		cl::Kernel kernel_1 = cl::Kernel(program, "total_Add");
		ExecuteKernal(queue, buffer_A, buffer_B, input_size, A, Mean, kernel_1, size,local_size,input_elements,prof_event,1);
		
		//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		
		cl::Kernel kernel_Max = cl::Kernel(program, "Maximum_Local");
		ExecuteKernal(queue, buffer_A, buffer_B, input_size, A, Mean, kernel_Max, size, local_size, input_elements, prof_event, 2);

		//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

		cl::Kernel kernel_Min = cl::Kernel(program, "Minimum_Local");
		ExecuteKernal(queue, buffer_A, buffer_B, input_size, A, Mean, kernel_Min, size, local_size, input_elements, prof_event, 3);
				
		//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueFillBuffer(buffer_Var, 0, 0, output_size);

		cl::Kernel kernel_Var = cl::Kernel(program, "Variance");
		kernel_Var.setArg(0, buffer_A);
		kernel_Var.setArg(1, buffer_Var);
		kernel_Var.setArg(2, (total / size));

		queue.enqueueNDRangeKernel(kernel_Var, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);
		queue.enqueueReadBuffer(buffer_Var, CL_TRUE, 0, input_size, &Var[0]);

		std::cout << "\tVariance = Complete" << std::endl;
		std::cout << "\t" << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << endl;
		std::cout << "\tKernel execution time[ns]:" << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		cout << "--------------------------------------------------------------------------------" << endl;
		
		//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &Var[0]);
		queue.enqueueFillBuffer(buffer_Dev, 0, 0, output_size);

		cl::Kernel kernel_Dev = cl::Kernel(program, "total_Add");
		kernel_Dev.setArg(0, buffer_A);
		kernel_Dev.setArg(1, buffer_Dev);
		kernel_Dev.setArg(2, cl::Local(local_size * sizeof(mytype)));

		queue.enqueueNDRangeKernel(kernel_Dev, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);
		queue.enqueueReadBuffer(buffer_Dev, CL_TRUE, 0, output_size, &Dev[0]);

		std::chrono::high_resolution_clock::time_point DevTime = std::chrono::high_resolution_clock::now();
		mytype stdDev = 0; for (int i = 0; i < nr_groups/2; i++) { stdDev += Dev[i]; }
		cout << "\tHost Time[ns]: " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - DevTime).count() << endl;
		stdDev = sqrt(stdDev / size);std::cout << "\tStandard Deviation = " << stdDev << std::endl;
		std::cout << "\t" << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << endl;
		std::cout << "\tKernel execution time[ns]:" << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		cout << "--------------------------------------------------------------------------------" << endl;

		//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		cout << "----The Program is about to begin sorting Floating point values! and may freeze until completion----" << endl;
		std::cout << "\t" << system("Pause");
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueFillBuffer(buffer_Sort, 0, 0, output_size);

		cl::Kernel kernel_Sort = cl::Kernel(program, "selection_sort");
		kernel_Sort.setArg(0, buffer_A);
		kernel_Sort.setArg(1, buffer_Sort);
		kernel_Sort.setArg(2, cl::Local(local_size * sizeof(mytype)));

		queue.enqueueNDRangeKernel(kernel_Sort, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &prof_event);
		queue.enqueueReadBuffer(buffer_Sort, CL_TRUE, 0, output_size, &Sort[0]);

		std::cout << "\tSorting Complete in Assending" << Sort[size-1] << std::endl;
		std::cout << "\t" << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << endl;
		std::cout << "\tKernel execution time[ns]:" << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		cout << "--------------------------------------------------------------------------------" << endl;

		std::cout << "\tMedian: " << Sort[Sort.size()/2] <<std::endl;
		std::cout << "\tLower: " << Sort[Sort.size() / 4] << std::endl;
		std::cout << "\tUpper: " << Sort[(Sort.size()*3) / 4] << std::endl;
		std::cout << "\tinterquartile Range: " << Sort[(Sort.size() * 3) / 4] - Sort[Sort.size() / 4] << std::endl;

		cout << "--------------------------------------------------------------------------------" << endl;
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	std::cout << "\t" << system("Pause");
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
