void addShapesF(float **base_s1, float **base_s2, float **base_s3, unsigned int r, unsigned int c);
void subtractShapesF(float **base_s1, float **base_s2, float **base_s3, unsigned int r, unsigned int c);
void crossShapesF(float **base_s1, float **base_s2, float **base_s3, unsigned int r, unsigned int c);
void divideShapesF(float **base_s1, float **base_s2, float **base_s3, unsigned int r, unsigned int c);
void dotMatricesF(const float *s1, const float *s2, float *s3, const unsigned int r, const unsigned int c,
                  const unsigned int c2);
void matVecF(float **base_s1, float **base_s2, float **base_s3, unsigned int r, unsigned int c);
float *createShapeF(const unsigned int n, const float fill_val);
void gpuInit();
void gpuClean();