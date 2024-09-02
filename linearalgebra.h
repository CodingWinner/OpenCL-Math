/*!
    @mainpage Linear Algebra For C

    @date 9/1/2024
    @author Ekansh Jain

    @details
    Basic functions to use matrices and vectors in C for things like A.I. and graphics. This project uses OpenCL for maximum efficiency

    @file linearalgebra.h
*/

/*!
    @page contents Table of Contents

    @tableofcontents Table of Contents

    @section general General

    @ref gpuInit()

    @ref gpuClean()

    @section vectors Vector Operations
    @ref MultiFOps

    @subsection funcs Functions
    @subsubsection groupFuncs Functions in Matrix and Vector Operations
    @ref addShapesF()

    @ref subtractShapesF()

    @ref crossShapesF()

    @ref divideShapesF()

    @ref createShapeF()

    @subsubsection otherFuncs Other Functions

    @ref matVecF()

    @section matrices Matrix Operations
    @ref MultiFOps

    @subsection mfuncs Functions
    @subsubsection groupMFuncs Functions in Matrix and Vector Operations
    @ref addShapesF()

    @ref subtractShapesF()

    @ref crossShapesF()

    @ref divideShapesF()

    @ref createShapeF()

    @subsubsection otherMFuncs Other Functions

    @ref dotMatricesF()

    @ref matVecF()

    @section kernels Operation Kernels
*/

/*!
    @page reference Function References

    @ref addShapesF()

    @ref createShapeF()

    @ref crossShapesF()

    @ref divideShapesF()

    @ref dotMatricesF()

    @ref gpuClean()

    @ref gpuInit()

    @ref matVecF()

    @ref subtractShapesF()
*/

typedef struct
{
    cl_kernel addFKernel;
    cl_kernel subtractFKernel;
    cl_kernel crossFKernel;
    cl_kernel divideFKernel;
    cl_kernel dotFKernel;
    cl_kernel matVecFkernel;
} Kernels;
typedef struct
{
    cl_event addFEvent;
    cl_event subtractFEvent;
    cl_event crossFEvent;
    cl_event divideFEvent;
    cl_event dotFEvent;
    cl_event matVecFEvent;
    cl_event s1Write;
    cl_event s2Write;
    cl_event s3Write;
} Events;
typedef struct
{
    cl_mem s1;
    cl_mem s2;
    cl_mem s3;
} Buffers;
typedef struct
{
    Kernels kernels;
    Events events;
    Buffers buffers;
    cl_platform_id platform;
    cl_context context;
    cl_device_id device;
    cl_command_queue queue;
    cl_program program;
    cl_int err;
} GPU;

/*!
    @defgroup MultiFOps Matrix and Vector Operations
    @brief This topic includes all the functions that can run for both matrices and vectors
    @{
*/

/*!
    @brief This functions is responsible for adding any of two shapes, matrices or vectors

    @details
    This function will take in 2 shapes and output the third shape which is the sum of the two shapes

    @param base_s1 This is the address to the first shape to be summed
    @param base_s2 This is the address to the second shape to be summed
    @param base_s3 This is the address to the third shape which will contain the sum of the other two shapes
    @param r This is the amount of rows in shapes 1 and 2
    @param c This is the amount of columns in shapes 1 and 2

    @remarks
    The shape at *base_s3 does not need to have the correct size allocated, it only needs to have some space allocated in the heap.
    This code will automatically allocate the right amount of space for the shape at *base_s3.
    This function does not have any error checking so you need to make sure that you give the right params to get the right output.
    This function is very similar to a few others see below

    @see subtractShapesF()
    @see crossShapesF()
    @see divideShapesF()
*/
void addShapesF(float **base_s1, float **base_s2, float **base_s3, unsigned int r, unsigned int c);
/*!
    @brief This function is responsible for subtracting any of two shapes, matrices or vectors

    @details This function will take in 2 shapes and output the third shape which will be the difference of the two shapes

    @param base_s1 This is the address to the first shape which will be subtracted from by the second shape
    @param base_s2 This is the address to the second shape which will subtract from the first shape
    @param base_s3 This is the address to the third shape which will contain the difference of the first two shapes
    @param r This is the amount of rows in shapes 1 and 2
    @param c This is the amount of columns in shapes 1 and 2

    @remarks
    The shape at *base_s3 does not need to have the correct size allocated, it only needs to have some space allocated in the heap.
    This code will automatically allocate the right amount of space for the shape at *base_s3
    This function does not have any error checking so you need to make sure that you give the right params to get the right output.
    This function is very similar to a few others see below

    @see addShapesF()
    @see crossShapesF()
    @see divideShapesF()
*/
void subtractShapesF(float **base_s1, float **base_s2, float **base_s3, unsigned int r, unsigned int c);
/*!
    @brief This function is responsible for either crossing two matrices or multiplying two vectors

    @details
    This function will take in 2 shapes and output the third shape which will be the cross or regular product of the two shapes

    @param base_s1 This is the address to the first shape which will be crossed or multiplied
    @param base_s2 This is the address to the second shape which will be crossed or multiplied
    @param base_s3 This is the address to the third shape which will contain the cross or regular product of the two shapes
    @param r This is the amount of rows in shapes 1 and 2
    @param c This is the amount of columns in shapes 1 and 2

    @remarks
    The shape at *base_s3 does not need to have the correct size allocated, it only needs to have some space allocated in the heap.
    This code will automatically allocate the right amount of space for the shape at *base_s3
    This function does not have any error checking so you need to make sure that you give the right params to get the right output.
    This function is very similar to a few others see below

    @see addShapesF()
    @see crossShapesF()
    @see divideShapesF()
*/
void crossShapesF(float **base_s1, float **base_s2, float **base_s3, unsigned int r, unsigned int c);
/*!
    @brief This function is responsible for dividing any of two shapes, matrices or vectors

    @details This function will take in 2 shapes and output the third shape which will be the quotient of the two shapes

    @param base_s1 This is the address to the first shape which will be the dividend
    @param base_s2 This is the address to the second shape which will be the divisor
    @param base_s3 This is the address to the third shape which will contain the quotient of the first two shapes
    @param r This is the amount of rows in shapes 1 and 2
    @param c This is the amount of columns in shapes 1 and 2

    @remarks
    The shape at *base_s3 does not need to have the correct size allocated, it only needs to have some space allocated in the heap.
    This code will automatically allocate the right amount of space for the shape at *base_s3
    This function does not have any error checking so you need to make sure that you give the right params to get the right output.
    This function is very similar to a few others see below

    @see addShapesF()
    @see subtractShapesF()
    @see crossShapesF()
*/
void divideShapesF(float **base_s1, float **base_s2, float **base_s3, unsigned int r, unsigned int c);
/*!
    @brief This function create either a matrix or vector

    @details
    Takes in two arguments and then uses them to create a shape which is given back to the user

    @param n This is the number of elements in the shape, for matrices it is row * columns, and for vectors it is just columns
    @param fill_val This is the default value that each element in the shape should be filled with

    @returns The shape with those requirements
*/
float *createShapeF(const unsigned int n, const float fill_val);

/*!
    @}
*/

/*!
    @brief This function calculates the dot product of 2 matrices

    @param s1 This is the first matrix to be used in the dot product
    @param s2 This is the second matrix to be used in the dot product
    @param s3 This is the third matrix which will contain the dot product
    @param r This is the number of rows in the first and third matrices
    @param c This is the number of columns in the first matrix and the number of rows in the second matrix
    @param c2 This is the number of columns in the second and third matrix

    @remarks
    There is no error checking in this function.
    The third shape must have correctly allocated space which is sizeof(float) * r * c2
    */
void dotMatricesF(const float *s1, const float *s2, float *s3, const unsigned int r, const unsigned int c, const unsigned int c2);
/*!
    @brief Multiplies a vector by a matrix

    @param base_s1 Address to the matrix which will multiply the vector
    @param base_s2 Address to the vector which will be multiplied by the matrix
    @param base_s3 Address to the vector which will store the result
    @param r Number of elements in the vector and the number of rows in the matrix
    @param c Number of columns in the matrix

    @remarks
    The vector at base_s3 must be allocated the correct space prior to this function being called.
    Their is no error checking in this function.
*/
void matVecF(float **base_s1, float **base_s2, float **base_s3, unsigned int r, unsigned int c);
/*!
    @brief Initializes the GPU struct. Must be called before any of the other functions
*/
void gpuInit();

/*!
    @brief Cleans up all the allocated memory in the GPU struct. Must be called before the program ends.
*/
void gpuClean();