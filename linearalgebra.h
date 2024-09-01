/*!
    @author Ekansh Jain
    @version 0.1.0
    @file Header file for vector and matrix operations in plain C.

    @mainpage Linear Algebra for C Documentation

    @details
    This projects goal was to create some support for using vectors and matrices in c for things like graphics and A.I.
    This is a very basic version custom fit to my device, but can be customized easily using the documentation as a guide.
    It uses OpenCL for parallel processing on the %GPU in order to optimize.

    @ref Contents "Table of Contents"
*/

/*!
    @page Contents Table of Contents

    @tableofcontents

    @section general General
    @section basics Basic Functions
    @ref FOperations "General Float Operations"
    @subsection functions Function List
    @ref addShapesF()

    @ref subtractShapesF()

    @ref crossShapesF()

    @ref divideShapesF()
    @section matrix Matrix Functions
    @ref FMatrixOperations "Float Matrix Specific Operations"
    @subsection functions Function List
    @ref dotMatricesF()
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
    @defgroup FOperations General Float Operations
    @brief Basic operations that work for float type matrices and vectors

    @details
    All the functions in this group are almost the same except for the fact that they have different operations

    @note
    There is no error checking for any of these functions. There will be unexpected results if the given params are not correct

    @ingroup FOperations
    @{

    @fn void addShapesF(float **base_s1, float **base_s2, float **base_s3, unsigned int r, unsigned int c)
    @brief Adds 2 shapes together and stores the sum in the third shape

    @details
    This function takes in 2 of the same shapes with their rows and columns, and outputs the addition of the shapes in the 3rd shape

    @param base_s1 This is the address to the first shape to be added
    @param base_s2 This is the address to the second shape to be added
    @param base_s3 This is the address to the shape which will contain the sum of the other 2 shapes
    @param r This is the number of rows in shapes 1, 2, and 3
    @param c This is the number of columns in shapes 1, 2, and 3

    @fn void subtractShapesF(float **base_s1, float **base_s2, float **base_s3, unsigned int r, unsigned int c)
    @brief Subtracts 2 shapes and stores the difference in the third shape

    @details
    This functions in 2 of the shapes with their rows and columns, then it outputs the difference of the shapes in the 3rd shape.

    @param base_s1 This is the address to the first shape, it will be the shape subtracted from
    @param base_s2 This is the shape that will be subtracted from the first shape
    @param base_s3 This is the shape that will contain the difference of the other two shapes
    @param r This is the number of rows in shapes 1, 2, and 3
    @param c This is the number of columns in shapes 1, 2, and 3

    @fn void crossShapesF(float **base_s1, float **base_s2, float **base_s3, unsigned int r, unsigned int c)
    @brief Uses the cross operation (x) on two shapes and stores the result in the 3rd shape

    @details
    This function can be used to cross 2 matrices or multiply 2 vectors

    @param base_s1 This is the address to the first shape to be crossed or multiplied
    @param base_s2 This is the address to the second shape to be crossed or multiplied
    @param base_s3 This is the address to the third shape which will store the cross product or regular product of the two shapes
    @param r This is the number of rows in shapes 1, 2, and 3
    @param c This is the number of columns in shapes 1, 2, and 3

    @fn void divideShapesF(float **base_s1, float **base_s2, float **base_s3, unsigned int r, unsigned int c)
    @brief Divides two shapes and stores the result in the third shape

    @details
    This function takes in 2 shapes and divides them into the third shape

    @param base_s1 This is the address to the first shape which is the dividend
    @param base_s2 This is the address to the second shape which is the divisor
    @param base_s3 This is the address to the third shape which will contain the quotient
    @param r This is the number of rows in shapes 1, 2, and 3
    @param c This is the number of columns in shapes 1, 2, and 3
    @}

    @defgroup FMatrixOperations Float Matrix Specific Operations
    @brief Operations that only exist for matrices

    @note
    There is no error checking in any of these functions. There will be unexpected results if the given params are not correct

    @ingroup FMatrixOperations
    @{
    @fn dotMatricesF(const float *s1, const float *s2, float *s3, const unsigned int r, const unsigned int c, const unsigned int c2)
    @brief Does the dot product of the first two matrices and stores in the third matrix

    @details
    Calculates the dot product of the first two matrices and stores it in the third matrix

    @param s1 The first matrix to be used in the dot product
    @param s2 The second matrix to be used in the dot product
    @param s3 The output matrix
    @param r The number of rows in \p s1 and \p s3
    @param c The number of columns in \p s1 and number of rows in \p s2
    @param c2 The number of columns in \p s2 and \p s3;

    @}
*/
void addShapesF(float **base_s1, float **base_s2, float **base_s3, unsigned int r, unsigned int c);
void subtractShapesF(float **base_s1, float **base_s2, float **base_s3, unsigned int r, unsigned int c);
void crossShapesF(float **base_s1, float **base_s2, float **base_s3, unsigned int r, unsigned int c);
void divideShapesF(float **base_s1, float **base_s2, float **base_s3, unsigned int r, unsigned int c);

void dotMatricesF(const float *s1, const float *s2, float *s3, const unsigned int r, const unsigned int c, const unsigned int c2);
void matVecF(float **base_s1, float **base_s2, float **base_s3, unsigned int r, unsigned int c);
float *createShapeF(const unsigned int n, const float fill_val);
void gpuInit();
void gpuClean();