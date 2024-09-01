// /*!
//     @author Ekansh Jain
//     @version 0.1.0
//     @file Code for supporting operations between vectors and matrices in plain C
// */

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
    @defgroup Operations General Operations
    @brief Basic operations across matrices and vectors
*/

/*!
    @ingroup Operations
    @{
*/

/*!
    @fn void addShapesF(float **base_s1, float **base_s2, float **base_s3, unsigned int r, unsigned int c)
    @brief Adds 2 shapes together

    @details
    This function takes in 2 shapes (Vectors or Matrices) with the rows and columns and outputs the addition of the shapes in the 3rd shape
    It only takes in shapes of type float

    @param base_s1 This is the address to the first shape to be added
    @param base_s2 This is the address to the second shape to be added
    @param base_s3 This is the address to the shape which will contain the sum of shapes at addresses \p base_s1 and \p base_s2
    @param r This is the number of rows in shapes 1, 2, and 3
    @param c This is the number of columns in shapes 1, 2, and 3
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