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

    @section matrices Matrix Operations

    @section kernels Operation Kernels
*/

/*!
    @page reference Function References
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

void addShapesF(float **base_s1, float **base_s2, float **base_s3, unsigned int r, unsigned int c);
void subtractShapesF(float **base_s1, float **base_s2, float **base_s3, unsigned int r, unsigned int c);
void crossShapesF(float **base_s1, float **base_s2, float **base_s3, unsigned int r, unsigned int c);
void divideShapesF(float **base_s1, float **base_s2, float **base_s3, unsigned int r, unsigned int c);
void dotMatricesF(const float *s1, const float *s2, float *s3, const unsigned int r, const unsigned int c, const unsigned int c2);
void matVecF(float **base_s1, float **base_s2, float **base_s3, unsigned int r, unsigned int c);
float *createShapeF(const unsigned int n, const float fill_val);
void gpuInit();
void gpuClean();