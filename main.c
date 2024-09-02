/*!
    @file main.c
*/

#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <linearalgebra.h>

/*!
    @brief Contains the source code for all the kernels that run on the gpu

    @details
    This kernel_code string variable has all the code required to perform the operations on any device of OpenCL.
    Each kernel function is specified by a __kernel tag in the beginning and ends with a line that only has a newline character.
    The __kernel tag indicates that this runs on a OpenCL used device. The ID for that device is stored in @ref GPU.device.
    Parameters that go into the kernels in here can have one of two potential starting tags (__global or __local).
    Params that have a __global tag in this code are buffers that transport memory from the CPU to the GPU. These have been used for arrays.
    Params that have a __local tag in this code are temporary memory spaces which have only been used for the last two kernels.
    Each work group has it's own __local memory space.
    Void is the return type which should stay void since you're not returning to anything if using a gpu kernel as in this situation.

    @note
    This variable should only be used for customizing/improving this code
*/
const char *kernel_code =
    "__kernel void addShapesF(__global const float *s1, __global const float *s2,\n"
    "                         __global float *s3, const unsigned int n)\n"
    "{\n"
    "    __private int index = get_global_id(0);\n"
    "    if (index < n)\n"
    "    {\n"
    "        s3[index] = s1[index] + s2[index];\n"
    "    }\n"
    "}\n"
    "\n"
    "__kernel void subtractShapesF(__global const float *s1,\n"
    "                              __global const float *s2, __global float *s3,\n"
    "                              const unsigned int n)\n"
    "{\n"
    "    __private int index = get_global_id(0);\n"
    "    if (index < n)\n"
    "    {\n"
    "        s3[index] = s1[index] - s2[index];\n"
    "    }\n"
    "}\n"
    "\n"
    "__kernel void crossShapesF(__global const float *s1, __global const float *s2,\n"
    "                           __global float *s3, const unsigned int n)\n"
    "{\n"
    "    __private int index = get_global_id(0);\n"
    "    if (index < n)\n"
    "    {\n"
    "        s3[index] = s1[index] * s2[index];\n"
    "    }\n"
    "}\n"
    "\n"
    "__kernel void divideShapesF(__global const float *s1, __global const float *s2,\n"
    "                            __global float *s3, const unsigned int n)\n"
    "{\n"
    "    __private int index = get_global_id(0);\n"
    "    if (index < n)\n"
    "    {\n"
    "        s3[index] = s1[index] / s2[index];\n"
    "    }\n"
    "}\n"
    "\n"
    "__kernel void dotMatricesF(__global const float *s1, __global const float *s2,\n\
                          __global float *s3, __local float *partial_sums,\n\
                          const unsigned int r, const unsigned int c, unsigned const int c2) {\n\
        __private const int row = get_global_id(0);\n\
        __private const int col2 = get_global_id(1);\n\
        __private const int col = get_local_id(0);\n\
        if (row < r && col < c && col2 < c2) {\n\
            partial_sums[col] = s1[row * c + col] * s2[col * c2 + col2];\n\
            barrier(CLK_LOCAL_MEM_FENCE);\n\
            if (col == 0) {\n\
                s3[row * c2 + col2] = 0;\n\
                for (int i = 0; i < c; i++) {\n\
                    s3[row * c2 + col2] += partial_sums[i];\n\
                }\n\
            }\n\
        }\n\
    }\n"
    "\n"
    "__kernel void MatrixFMulVecF(__global const float *m, __global const float *v,\n\
                           __global float *out, __local float *partial_sums,\n\
                           const unsigned int r, const unsigned int c)\n\
{\n\
    __private const unsigned int row = get_global_id(0);\n\
    __private const unsigned int col = get_global_id(1);\n\
    if (row < r && col < c)\n\
    {\n\
        partial_sums[col] = m[row * c + col] * v[row];\n\
        barrier(CLK_LOCAL_MEM_FENCE);\n\
        if (col == 0)\n\
        {\n\
            out[row] = 0;\n\
            for (int i = 0; i < c; i++)\n\
            {\n\
                out[row] += partial_sums[i];\n\
            }\n\
        }\n\
    }\n\
}\n\
";

/*!
    @brief The variable used to access GPU information like the kernels or deviceID

    @note This should only be used when creating your own gpu kernel function
*/
GPU gpu;

/*!
    @brief Function to check for any OpenCL error and output the code
*/
void checkError()
{
    if (gpu.err != CL_SUCCESS)
    {
        printf("Error code: %i \n", gpu.err);
        exit(1);
    }
}
void addShapesF(float **base_s1, float **base_s2, float **base_s3, unsigned int r, unsigned int c)
{
    const size_t old_size = sizeof(float) * r * c;
    if (r == 1)
    {
        if (c < 32)
        {
            c = 32;
        }
        else if (c < 64 && c > 32)
        {
            c = 64;
        }
        else if (c < 128 && c > 64)
        {
            c = 128;
        }
        else if (c < 256 && c > 128)
        {
            c = 256;
        }
    }
    else if (c == 1)
    {
        if (r < 32)
        {
            r = 32;
        }
        else if (r < 64 && r > 32)
        {
            r = 64;
        }
        else if (r < 128 && r > 64)
        {
            r = 128;
        }
        else if (r < 256 && r > 128)
        {
            r = 256;
        }
    }
    else if (r < 32)
    {
        r = 32;
    }
    else if (c < 32)
    {
        c = 32;
    }
    else
    {
        if (c < 32)
        {
            c = 32;
        }
        else if (c < 64 && c > 32)
        {
            c = 64;
        }
        else if (c < 128 && c > 64)
        {
            c = 128;
        }
        else if (c < 256 && c > 128)
        {
            c = 256;
        }
        if (r < 32)
        {
            r = 32;
        }
        else if (r < 64 && r > 32)
        {
            r = 64;
        }
        else if (r < 128 && r > 64)
        {
            r = 128;
        }
        else if (r < 256 && r > 128)
        {
            r = 256;
        }
    }
    const size_t size = sizeof(float) * r * c;
    *base_s1 = realloc(*base_s1, size);
    *base_s2 = realloc(*base_s2, size);
    *base_s3 = realloc(*base_s3, size);
    gpu.buffers.s1 = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, size, NULL, &gpu.err);
    gpu.buffers.s2 = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, size, NULL, &gpu.err);
    gpu.buffers.s3 = clCreateBuffer(gpu.context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, size, NULL, &gpu.err);
    gpu.err = clEnqueueWriteBuffer(gpu.queue, gpu.buffers.s1, CL_TRUE, 0, size, *base_s1, 0, NULL, &gpu.events.s1Write);
    gpu.err = clEnqueueWriteBuffer(gpu.queue, gpu.buffers.s2, CL_TRUE, 0, size, *base_s2, 0, NULL, &gpu.events.s2Write);
    const cl_event bufferEvents[2] = {gpu.events.s1Write, gpu.events.s2Write};
    const unsigned int vals = r * c;
    gpu.err = clWaitForEvents(2, bufferEvents);
    gpu.err = clSetKernelArg(gpu.kernels.addFKernel, 0, sizeof(cl_mem), &gpu.buffers.s1);
    gpu.err = clSetKernelArg(gpu.kernels.addFKernel, 1, sizeof(cl_mem), &gpu.buffers.s2);
    gpu.err = clSetKernelArg(gpu.kernels.addFKernel, 2, sizeof(cl_mem), &gpu.buffers.s3);
    gpu.err = clSetKernelArg(gpu.kernels.addFKernel, 3, sizeof(const unsigned int), &vals);
    if (r == 1 || c == 1)
    {
        size_t globalSize[1];
        if (r == 1)
        {
            globalSize[0] = c;
        }
        else
        {
            globalSize[0] = r;
        }
        const size_t localSize[1] = {32};
        gpu.err = clEnqueueNDRangeKernel(gpu.queue, gpu.kernels.addFKernel, 1, NULL, globalSize, localSize, 0, NULL, &gpu.events.addFEvent);
    }
    else
    {
        const size_t globalSize[2] = {r, c};
        const size_t localSize[2] = {32, 32};
        gpu.err = clEnqueueNDRangeKernel(gpu.queue, gpu.kernels.addFKernel, 1, NULL, globalSize, localSize, 0, NULL, &gpu.events.addFEvent);
    }
    gpu.err = clWaitForEvents(1, &gpu.events.addFEvent);
    gpu.err = clEnqueueReadBuffer(gpu.queue, gpu.buffers.s3, CL_TRUE, 0, size, *base_s3, 0, NULL, &gpu.events.s3Write);
    gpu.err = clWaitForEvents(1, &gpu.events.s3Write);

    clReleaseMemObject(gpu.buffers.s1);
    clReleaseMemObject(gpu.buffers.s2);
    clReleaseMemObject(gpu.buffers.s3);

    *base_s1 = realloc(*base_s1, old_size);
    *base_s2 = realloc(*base_s2, old_size);
    *base_s3 = realloc(*base_s3, old_size);
}
void subtractShapesF(float **base_s1, float **base_s2, float **base_s3, unsigned int r, unsigned int c)
{
    const size_t old_size = sizeof(float) * r * c;
    if (r == 1)
    {
        if (c < 32)
        {
            c = 32;
        }
        else if (c < 64 && c > 32)
        {
            c = 64;
        }
        else if (c < 128 && c > 64)
        {
            c = 128;
        }
        else if (c < 256 && c > 128)
        {
            c = 256;
        }
    }
    else if (c == 1)
    {
        if (r < 32)
        {
            r = 32;
        }
        else if (r < 64 && r > 32)
        {
            r = 64;
        }
        else if (r < 128 && r > 64)
        {
            r = 128;
        }
        else if (r < 256 && r > 128)
        {
            r = 256;
        }
    }
    else if (r < 32)
    {
        r = 32;
    }
    else if (c < 32)
    {
        c = 32;
    }
    else
    {
        if (c < 32)
        {
            c = 32;
        }
        else if (c < 64 && c > 32)
        {
            c = 64;
        }
        else if (c < 128 && c > 64)
        {
            c = 128;
        }
        else if (c < 256 && c > 128)
        {
            c = 256;
        }
        if (r < 32)
        {
            r = 32;
        }
        else if (r < 64 && r > 32)
        {
            r = 64;
        }
        else if (r < 128 && r > 64)
        {
            r = 128;
        }
        else if (r < 256 && r > 128)
        {
            r = 256;
        }
    }
    const size_t size = sizeof(float) * r * c;
    *base_s1 = realloc(*base_s1, size);
    *base_s2 = realloc(*base_s2, size);
    *base_s3 = realloc(*base_s3, size);
    gpu.buffers.s1 = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, size, NULL, &gpu.err);
    gpu.buffers.s2 = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, size, NULL, &gpu.err);
    gpu.buffers.s3 = clCreateBuffer(gpu.context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, size, NULL, &gpu.err);
    gpu.err = clEnqueueWriteBuffer(gpu.queue, gpu.buffers.s1, CL_TRUE, 0, size, *base_s1, 0, NULL, &gpu.events.s1Write);
    gpu.err = clEnqueueWriteBuffer(gpu.queue, gpu.buffers.s2, CL_TRUE, 0, size, *base_s2, 0, NULL, &gpu.events.s2Write);
    const cl_event bufferEvents[2] = {gpu.events.s1Write, gpu.events.s2Write};
    const unsigned int vals = r * c;
    gpu.err = clWaitForEvents(2, bufferEvents);
    gpu.err = clSetKernelArg(gpu.kernels.subtractFKernel, 0, sizeof(cl_mem), &gpu.buffers.s1);
    gpu.err = clSetKernelArg(gpu.kernels.subtractFKernel, 1, sizeof(cl_mem), &gpu.buffers.s2);
    gpu.err = clSetKernelArg(gpu.kernels.subtractFKernel, 2, sizeof(cl_mem), &gpu.buffers.s3);
    gpu.err = clSetKernelArg(gpu.kernels.subtractFKernel, 3, sizeof(const unsigned int), &vals);
    if (r == 1 || c == 1)
    {
        size_t globalSize[1];
        if (r == 1)
        {
            globalSize[0] = c;
        }
        else
        {
            globalSize[0] = r;
        }
        const size_t localSize[1] = {32};
        gpu.err = clEnqueueNDRangeKernel(gpu.queue, gpu.kernels.subtractFKernel, 1, NULL, globalSize, localSize, 0, NULL, &gpu.events.subtractFEvent);
    }
    else
    {
        const size_t globalSize[2] = {r, c};
        const size_t localSize[2] = {32, 32};
        gpu.err = clEnqueueNDRangeKernel(gpu.queue, gpu.kernels.subtractFKernel, 1, NULL, globalSize, localSize, 0, NULL, &gpu.events.subtractFEvent);
    }
    gpu.err = clWaitForEvents(1, &gpu.events.subtractFEvent);
    gpu.err = clEnqueueReadBuffer(gpu.queue, gpu.buffers.s3, CL_TRUE, 0, size, *base_s3, 0, NULL, &gpu.events.s3Write);
    gpu.err = clWaitForEvents(1, &gpu.events.s3Write);

    clReleaseMemObject(gpu.buffers.s1);
    clReleaseMemObject(gpu.buffers.s2);
    clReleaseMemObject(gpu.buffers.s3);
    *base_s1 = realloc(*base_s1, old_size);
    *base_s2 = realloc(*base_s2, old_size);
    *base_s3 = realloc(*base_s3, old_size);
}
void crossShapesF(float **base_s1, float **base_s2, float **base_s3, unsigned int r, unsigned int c)
{
    const size_t old_size = sizeof(float) * r * c;
    if (r == 1)
    {
        if (c < 32)
        {
            c = 32;
        }
        else if (c < 64 && c > 32)
        {
            c = 64;
        }
        else if (c < 128 && c > 64)
        {
            c = 128;
        }
        else if (c < 256 && c > 128)
        {
            c = 256;
        }
    }
    else if (c == 1)
    {
        if (r < 32)
        {
            r = 32;
        }
        else if (r < 64 && r > 32)
        {
            r = 64;
        }
        else if (r < 128 && r > 64)
        {
            r = 128;
        }
        else if (r < 256 && r > 128)
        {
            r = 256;
        }
    }
    else if (r < 32)
    {
        r = 32;
    }
    else if (c < 32)
    {
        c = 32;
    }
    else
    {
        if (c < 32)
        {
            c = 32;
        }
        else if (c < 64 && c > 32)
        {
            c = 64;
        }
        else if (c < 128 && c > 64)
        {
            c = 128;
        }
        else if (c < 256 && c > 128)
        {
            c = 256;
        }
        if (r < 32)
        {
            r = 32;
        }
        else if (r < 64 && r > 32)
        {
            r = 64;
        }
        else if (r < 128 && r > 64)
        {
            r = 128;
        }
        else if (r < 256 && r > 128)
        {
            r = 256;
        }
    }
    const size_t size = sizeof(float) * r * c;
    *base_s1 = realloc(*base_s1, size);
    *base_s2 = realloc(*base_s2, size);
    *base_s3 = realloc(*base_s3, size);
    gpu.buffers.s1 = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, size, NULL, &gpu.err);
    gpu.buffers.s2 = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, size, NULL, &gpu.err);
    gpu.buffers.s3 = clCreateBuffer(gpu.context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, size, NULL, &gpu.err);
    gpu.err = clEnqueueWriteBuffer(gpu.queue, gpu.buffers.s1, CL_TRUE, 0, size, *base_s1, 0, NULL, &gpu.events.s1Write);
    gpu.err = clEnqueueWriteBuffer(gpu.queue, gpu.buffers.s2, CL_TRUE, 0, size, *base_s2, 0, NULL, &gpu.events.s2Write);
    const cl_event bufferEvents[2] = {gpu.events.s1Write, gpu.events.s2Write};
    const unsigned int vals = r * c;
    gpu.err = clWaitForEvents(2, bufferEvents);
    gpu.err = clSetKernelArg(gpu.kernels.crossFKernel, 0, sizeof(cl_mem), &gpu.buffers.s1);
    gpu.err = clSetKernelArg(gpu.kernels.crossFKernel, 1, sizeof(cl_mem), &gpu.buffers.s2);
    gpu.err = clSetKernelArg(gpu.kernels.crossFKernel, 2, sizeof(cl_mem), &gpu.buffers.s3);
    gpu.err = clSetKernelArg(gpu.kernels.crossFKernel, 3, sizeof(const unsigned int), &vals);
    if (r == 1 || c == 1)
    {
        size_t globalSize[1];
        if (r == 1)
        {
            globalSize[0] = c;
        }
        else
        {
            globalSize[0] = r;
        }
        const size_t localSize[1] = {32};
        gpu.err = clEnqueueNDRangeKernel(gpu.queue, gpu.kernels.crossFKernel, 1, NULL, globalSize, localSize, 0, NULL, &gpu.events.crossFEvent);
    }
    else
    {
        const size_t globalSize[2] = {r, c};
        const size_t localSize[2] = {32, 32};
        gpu.err = clEnqueueNDRangeKernel(gpu.queue, gpu.kernels.crossFKernel, 1, NULL, globalSize, localSize, 0, NULL, &gpu.events.crossFEvent);
    }
    gpu.err = clWaitForEvents(1, &gpu.events.crossFEvent);
    gpu.err = clEnqueueReadBuffer(gpu.queue, gpu.buffers.s3, CL_TRUE, 0, size, *base_s3, 0, NULL, &gpu.events.s3Write);
    gpu.err = clWaitForEvents(1, &gpu.events.s3Write);

    clReleaseMemObject(gpu.buffers.s1);
    clReleaseMemObject(gpu.buffers.s2);
    clReleaseMemObject(gpu.buffers.s3);
    *base_s1 = realloc(*base_s1, old_size);
    *base_s2 = realloc(*base_s2, old_size);
    *base_s3 = realloc(*base_s3, old_size);
}
void divideShapesF(float **base_s1, float **base_s2, float **base_s3, unsigned int r, unsigned int c)
{
    const size_t old_size = sizeof(float) * r * c;
    if (r == 1)
    {
        if (c < 32)
        {
            c = 32;
        }
        else if (c < 64 && c > 32)
        {
            c = 64;
        }
        else if (c < 128 && c > 64)
        {
            c = 128;
        }
        else if (c < 256 && c > 128)
        {
            c = 256;
        }
    }
    else if (c == 1)
    {
        if (r < 32)
        {
            r = 32;
        }
        else if (r < 64 && r > 32)
        {
            r = 64;
        }
        else if (r < 128 && r > 64)
        {
            r = 128;
        }
        else if (r < 256 && r > 128)
        {
            r = 256;
        }
    }
    else if (r < 32)
    {
        r = 32;
    }
    else if (c < 32)
    {
        c = 32;
    }
    else
    {
        if (c < 32)
        {
            c = 32;
        }
        else if (c < 64 && c > 32)
        {
            c = 64;
        }
        else if (c < 128 && c > 64)
        {
            c = 128;
        }
        else if (c < 256 && c > 128)
        {
            c = 256;
        }
        if (r < 32)
        {
            r = 32;
        }
        else if (r < 64 && r > 32)
        {
            r = 64;
        }
        else if (r < 128 && r > 64)
        {
            r = 128;
        }
        else if (r < 256 && r > 128)
        {
            r = 256;
        }
    }
    const size_t size = sizeof(float) * r * c;
    *base_s1 = realloc(*base_s1, size);
    *base_s2 = realloc(*base_s2, size);
    *base_s3 = realloc(*base_s3, size);
    gpu.buffers.s1 = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, size, NULL, &gpu.err);
    gpu.buffers.s2 = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, size, NULL, &gpu.err);
    gpu.buffers.s3 = clCreateBuffer(gpu.context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, size, NULL, &gpu.err);
    gpu.err = clEnqueueWriteBuffer(gpu.queue, gpu.buffers.s1, CL_TRUE, 0, size, *base_s1, 0, NULL, &gpu.events.s1Write);
    gpu.err = clEnqueueWriteBuffer(gpu.queue, gpu.buffers.s2, CL_TRUE, 0, size, *base_s2, 0, NULL, &gpu.events.s2Write);
    const cl_event bufferEvents[2] = {gpu.events.s1Write, gpu.events.s2Write};
    const unsigned int vals = r * c;
    gpu.err = clWaitForEvents(2, bufferEvents);
    gpu.err = clSetKernelArg(gpu.kernels.divideFKernel, 0, sizeof(cl_mem), &gpu.buffers.s1);
    gpu.err = clSetKernelArg(gpu.kernels.divideFKernel, 1, sizeof(cl_mem), &gpu.buffers.s2);
    gpu.err = clSetKernelArg(gpu.kernels.divideFKernel, 2, sizeof(cl_mem), &gpu.buffers.s3);
    gpu.err = clSetKernelArg(gpu.kernels.divideFKernel, 3, sizeof(const unsigned int), &vals);
    if (r == 1 || c == 1)
    {
        size_t globalSize[1];
        if (r == 1)
        {
            globalSize[0] = c;
        }
        else
        {
            globalSize[0] = r;
        }
        const size_t localSize[1] = {32};
        gpu.err = clEnqueueNDRangeKernel(gpu.queue, gpu.kernels.divideFKernel, 1, NULL, globalSize, localSize, 0, NULL, &gpu.events.divideFEvent);
    }
    else
    {
        const size_t globalSize[2] = {r, c};
        const size_t localSize[2] = {32, 32};
        gpu.err = clEnqueueNDRangeKernel(gpu.queue, gpu.kernels.divideFKernel, 1, NULL, globalSize, localSize, 0, NULL, &gpu.events.divideFEvent);
    }
    gpu.err = clWaitForEvents(1, &gpu.events.divideFEvent);
    gpu.err = clEnqueueReadBuffer(gpu.queue, gpu.buffers.s3, CL_TRUE, 0, size, *base_s3, 0, NULL, &gpu.events.s3Write);
    gpu.err = clWaitForEvents(1, &gpu.events.s3Write);

    clReleaseMemObject(gpu.buffers.s1);
    clReleaseMemObject(gpu.buffers.s2);
    clReleaseMemObject(gpu.buffers.s3);
    *base_s1 = realloc(*base_s1, old_size);
    *base_s2 = realloc(*base_s2, old_size);
    *base_s3 = realloc(*base_s3, old_size);
}
void dotMatricesF(const float *s1, const float *s2, float *s3, const unsigned int r, const unsigned int c, const unsigned int c2)
{
    const size_t size1 = sizeof(float) * r * c;
    const size_t size2 = sizeof(float) * c * c2;
    const size_t size3 = sizeof(float) * r * c2;
    gpu.buffers.s1 = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, size1, NULL, &gpu.err);
    checkError();
    gpu.buffers.s2 = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, size2, NULL, &gpu.err);
    checkError();
    gpu.buffers.s3 = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, size3, NULL, &gpu.err);
    checkError();
    gpu.err = clEnqueueWriteBuffer(gpu.queue, gpu.buffers.s1, CL_TRUE, 0, size1, s1, 0, NULL, &gpu.events.s1Write);
    checkError();
    gpu.err = clEnqueueWriteBuffer(gpu.queue, gpu.buffers.s2, CL_TRUE, 0, size2, s2, 0, NULL, &gpu.events.s2Write);
    checkError();
    cl_event bufferEvents[2] = {gpu.events.s1Write, gpu.events.s2Write};
    gpu.err = clWaitForEvents(2, bufferEvents);
    checkError();
    gpu.err = clSetKernelArg(gpu.kernels.dotFKernel, 0, sizeof(cl_mem), &gpu.buffers.s1);
    checkError();
    gpu.err = clSetKernelArg(gpu.kernels.dotFKernel, 1, sizeof(cl_mem), &gpu.buffers.s2);
    checkError();
    gpu.err = clSetKernelArg(gpu.kernels.dotFKernel, 2, sizeof(cl_mem), &gpu.buffers.s3);
    checkError();
    gpu.err = clSetKernelArg(gpu.kernels.dotFKernel, 3, sizeof(float) * c, NULL);
    checkError();
    gpu.err = clSetKernelArg(gpu.kernels.dotFKernel, 4, sizeof(const unsigned int), &r);
    checkError();
    gpu.err = clSetKernelArg(gpu.kernels.dotFKernel, 5, sizeof(const unsigned int), &c);
    checkError();
    gpu.err = clSetKernelArg(gpu.kernels.dotFKernel, 6, sizeof(const unsigned int), &c2);
    checkError();
    const size_t global_work_size[3] = {r, c, c2};
    const size_t local_work_size[3] = {1, c, 1};
    gpu.err = clEnqueueNDRangeKernel(gpu.queue, gpu.kernels.dotFKernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &gpu.events.dotFEvent);
    checkError();
    gpu.err = clWaitForEvents(1, &gpu.events.dotFEvent);
    checkError();
    gpu.err = clEnqueueReadBuffer(gpu.queue, gpu.buffers.s3, CL_TRUE, 0, size3, s3, 0, NULL, &gpu.events.s3Write);
    checkError();
    gpu.err = clWaitForEvents(1, &gpu.events.s3Write);
    checkError();

    clReleaseMemObject(gpu.buffers.s1);
    clReleaseMemObject(gpu.buffers.s2);
    clReleaseMemObject(gpu.buffers.s3);
}
void matVecF(float **base_s1, float **base_s2, float **base_s3, unsigned int r, unsigned int c)
{
    const size_t matrix_size = sizeof(float) * r * c;
    const size_t vector_size = sizeof(float) * r;
    gpu.buffers.s1 = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, matrix_size, NULL, &gpu.err);
    gpu.buffers.s2 = clCreateBuffer(gpu.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, vector_size, NULL, &gpu.err);
    gpu.buffers.s3 = clCreateBuffer(gpu.context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, vector_size, NULL, &gpu.err);
    gpu.err = clEnqueueWriteBuffer(gpu.queue, gpu.buffers.s1, CL_TRUE, 0, matrix_size, *base_s1, 0, NULL, &gpu.events.s1Write);
    gpu.err = clEnqueueWriteBuffer(gpu.queue, gpu.buffers.s2, CL_TRUE, 0, vector_size, *base_s2, 0, NULL, &gpu.events.s2Write);
    const cl_event bufferEvents[2] = {gpu.events.s1Write, gpu.events.s2Write};
    gpu.err = clWaitForEvents(2, bufferEvents);
    gpu.err = clSetKernelArg(gpu.kernels.matVecFkernel, 0, sizeof(cl_mem), &gpu.buffers.s1);
    gpu.err = clSetKernelArg(gpu.kernels.matVecFkernel, 1, sizeof(cl_mem), &gpu.buffers.s2);
    gpu.err = clSetKernelArg(gpu.kernels.matVecFkernel, 2, sizeof(cl_mem), &gpu.buffers.s3);
    gpu.err = clSetKernelArg(gpu.kernels.matVecFkernel, 3, sizeof(float) * c, NULL);
    gpu.err = clSetKernelArg(gpu.kernels.matVecFkernel, 4, sizeof(const unsigned int), &r);
    gpu.err = clSetKernelArg(gpu.kernels.matVecFkernel, 5, sizeof(const unsigned int), &c);
    const size_t global_work_size[2] = {r, c};
    const size_t local_work_size[2] = {1, c};
    gpu.err = clEnqueueNDRangeKernel(gpu.queue, gpu.kernels.matVecFkernel, 2, NULL, global_work_size, local_work_size, 0, NULL, &gpu.events.matVecFEvent);
    gpu.err = clWaitForEvents(1, &gpu.events.matVecFEvent);
    gpu.err = clEnqueueReadBuffer(gpu.queue, gpu.buffers.s3, CL_TRUE, 0, vector_size, *base_s3, 0, NULL, &gpu.events.s3Write);
    gpu.err = clWaitForEvents(1, &gpu.events.s3Write);

    clReleaseMemObject(gpu.buffers.s1);
    clReleaseMemObject(gpu.buffers.s2);
    clReleaseMemObject(gpu.buffers.s3);
}
float *createShapeF(const unsigned int n, const float fill_val)
{
    size_t size = sizeof(float) * n;
    float *s1 = malloc(size);
    for (int i = 0; i < n; i++)
    {
        s1[i] = fill_val;
    }
    return s1;
}
void gpuInit()
{
    gpu.err = clGetPlatformIDs(1, &gpu.platform, NULL);
    gpu.err = clGetDeviceIDs(gpu.platform, CL_DEVICE_TYPE_GPU, 1, &gpu.device, NULL);
    gpu.context = clCreateContext(0, 1, &gpu.device, NULL, NULL, &gpu.err);
    gpu.queue = clCreateCommandQueue(gpu.context, gpu.device, CL_QUEUE_PROFILING_ENABLE, &gpu.err);
    gpu.program = clCreateProgramWithSource(gpu.context, 1, (const char **)&kernel_code, NULL, &gpu.err);
    clBuildProgram(gpu.program, 0, NULL, NULL, NULL, NULL);
    gpu.kernels.addFKernel = clCreateKernel(gpu.program, "addShapesF", &gpu.err);
    gpu.kernels.subtractFKernel = clCreateKernel(gpu.program, "subtractShapesF", &gpu.err);
    gpu.kernels.crossFKernel = clCreateKernel(gpu.program, "crossShapesF", &gpu.err);
    gpu.kernels.divideFKernel = clCreateKernel(gpu.program, "divideShapesF", &gpu.err);
    gpu.kernels.dotFKernel = clCreateKernel(gpu.program, "dotMatricesF", &gpu.err);
    gpu.kernels.matVecFkernel = clCreateKernel(gpu.program, "MatrixFMulVecF", &gpu.err);
}
void gpuClean()
{
    clReleaseKernel(gpu.kernels.addFKernel);
    clReleaseKernel(gpu.kernels.subtractFKernel);
    clReleaseKernel(gpu.kernels.crossFKernel);
    clReleaseKernel(gpu.kernels.divideFKernel);
    clReleaseKernel(gpu.kernels.dotFKernel);
    clReleaseKernel(gpu.kernels.matVecFkernel);
    clReleaseEvent(gpu.events.addFEvent);
    clReleaseEvent(gpu.events.crossFEvent);
    clReleaseEvent(gpu.events.subtractFEvent);
    clReleaseEvent(gpu.events.crossFEvent);
    clReleaseEvent(gpu.events.dotFEvent);
    clReleaseEvent(gpu.events.matVecFEvent);
    clReleaseEvent(gpu.events.s1Write);
    clReleaseEvent(gpu.events.s2Write);
    clReleaseEvent(gpu.events.s3Write);
    clReleaseProgram(gpu.program);
    clReleaseCommandQueue(gpu.queue);
    clReleaseContext(gpu.context);
    clReleaseDevice(gpu.device);
}
