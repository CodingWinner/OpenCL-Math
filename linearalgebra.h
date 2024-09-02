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

    @section vectors Vector Operations
    @ref MultiFOps

    @subsection funcs Functions
    @ref addShapesF()

    @ref subtractShapesF()

    @ref crossShapesF()

    @ref divideShapesF()

    @ref createShapeF()

    @section matrices Matrix Operations
    @ref MultiFOps

    @ref dotMatricesF()

    @subsection mfuncs Functions
    @ref addShapesF()

    @ref subtractShapesF()

    @ref crossShapesF()

    @ref divideShapesF()

    @ref createShapeF()

    @ref dotMatricesF()

    @section kernels Operation Kernels
*/

/*!
    @page reference Function References

    @ref addShapesF()

    @ref subtractShapesF()

    @ref createShapeF()

    @ref crossShapesF()

    @ref divideShapesF()
*/

/*!
    @brief %GPU kernels to perform operations
    @details
    A list of the kernels that work on the gpu to perform operations for vectors and matrices.

    @note
    Only use when adding kernels.
*/
typedef struct
{
    /*! @brief Variable to store the kernel responsible for adding two shapes*/
    cl_kernel addFKernel;
    /*! @brief Variable to store the kernel responsible for subtracting two shapes*/
    cl_kernel subtractFKernel;
    /*! @brief Variable to store the kernel responsible for crossing two matrices or multiplying two vectors*/
    cl_kernel crossFKernel;
    /*! @brief Variable to store the kernel responsible for dividing two shapes*/
    cl_kernel divideFKernel;
    /*! @brief Variable to store the kernel responsible for getting the dot prodcut of two matrices*/
    cl_kernel dotFKernel;
    /*! @brief Variable to store the kernel responsible for multiplying a vector by a matrix*/
    cl_kernel matVecFkernel;
} Kernels;
/*!
    @brief Contains a list of events that occur in functions throughout the code

    @details
    An event is basically a way to track if an OpenCL functions is done or not.
    This contains a list of all the events that occur that need to be tracked in order for proper code execution.

    @note
    Add/Remove events when adding to this code or customizing
*/
typedef struct
{
    /*!
        @brief Event for addition of shapes

        @details
        This event monitors when the addition kernel is done in order to make sure that s3 in @ref addShapesF() is getting filled with the correct values.
        It also is responsible for making sure the kernel isn't being asked to run with two different params at the same time as that can result in unexpected behavior.
    */
    cl_event addFEvent;
    /*!
        @brief Event for subtraction of shapes

        @details
        This event monitors when the subtraction kernel is done in order to make sure that s3 in @ref subtractShapesF() is getting filled with the correct values.
        It also is responsible for making sure the kernel isn't being asked to run with two different params at the same time as that can result in unexpected behavior.
    */
    cl_event subtractFEvent;
    /*!
        @brief Event for crossing or multiplication of shapes

        @details
        This event monitors when the crossing kernel is done in order to make sure that s3 in @ref crossShapesF() is getting filled with the correct values.
        It also is responsible for making sure the kernel isn't being asked to run with two different params at the same time as that can result in unexpected behavior.
    */
    cl_event crossFEvent;
    /*!
        @brief Event for division of shapes

        @details
        This event monitors when the divide kernel is done in order to make sure that s3 in @ref divideShapesF() is getting filled with the correct values.
        It also is responsible for making sure the kernel isn't being asked to run with two different params at the same time as that can result in unexpected behavior.
    */
    cl_event divideFEvent;
    /*!
        @brief Event for calculating the dot product of 2 matrices

        @details
        This event monitors when the dot kernel is done in order to make sure that s3 in @ref dotMatricesF() is getting filled with the correct values.
        It also is responsible for making sure the kernel isn't being asked to run with two different params at the same time as that can result in unexpected behavior.
    */
    cl_event dotFEvent;
    /*!
        @brief Event for kernel multiplying a vector by a matrix

        @details
        This event monitors when the matVec kernel is done in order to make sure that s3 in @ref matVecF() is getting filled with the correct values.
        It also is responsible for making sure the kernel isn't being asked to run with two different params at the same time as that can result in unexpected behavior.
    */
    cl_event matVecFEvent;
    /*!
        @brief This event monitors writing data to the s1 buffer
    */
    cl_event s1Write;
    /*!
        @brief This event monitors writing data to the s2 buffer or storing results from the s2 buffer
    */
    cl_event s2Write;
    /*!
        @brief This event monitors writing data to s3 to store results
    */
    cl_event s3Write;
} Events;
/*!
    @brief Structure for storing buffers which are used in various functions

    @details
    A buffer is basically a way to take an array from CPU memory to GPU memory

    @note
    Only add more buffers if adding operations that involve more than 3 arrays
*/
typedef struct
{
    /*!
        @brief Buffer for storing shape 1
    */
    cl_mem s1;
    /*!
        @brief Buffer for storing shape 2
    */
    cl_mem s2;
    /*!
        @brief Buffer for storing shape 3
    */
    cl_mem s3;
} Buffers;
/*!
    @brief Contains general information about the %GPU

    @details
    This struct contains the Kernels, Events, and Buffers struct as well as basic information about the %GPU in order for OpenCL to work

    @note This should really never be changed
*/
typedef struct
{
    /*! @brief The Kernels struct to use the kernels used in the program*/
    Kernels kernels;
    /*! @brief The Events struct to use events in the program*/
    Events events;
    /*! @brief The Buffers struct to use buffers in the program*/
    Buffers buffers;
    /*! @brief Contains the platform which is something OpenCL needs*/
    cl_platform_id platform;
    /*! @brief Contains the context which is also something OpenCL needs*/
    cl_context context;
    /*! @brief Contains the device ID, in this case the id of the %GPU to use in OpenCL functions*/
    cl_device_id device;
    /*! @brief Contains the command queue for running the kernels or writing to buffers*/
    cl_command_queue queue;
    /*! @brief Contains all the source code for the kernels: @ref kernel_code*/
    cl_program program;
    /*! @brief The variable to store errors from OpenCL functions*/
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
void gpuInit();
void gpuClean();