#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define N 1000
#define EPSILON 1e-2

// Allocate matrix
float** allocMatrix() {
    float** mat = (float**)malloc(N * sizeof(float*));
    for (int i = 0; i < N; i++)
        mat[i] = (float*)malloc(N * sizeof(float));
    return mat;
}

// Initialize matrix A
void initMatrix(float** A) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = (i + j) % 10 + 1;
}

// Verify LU ≈ A
int verifyLU(float** A, float** L, float** U) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0;
            for (int k = 0; k < N; k++)
                sum += L[i][k] * U[k][j];
            if (fabs(sum - A[i][j]) > EPSILON) {
                printf("Mismatch at A[%d][%d]: got %.2f, expected %.2f\n",
                       i, j, sum, A[i][j]);
                return 0;
            }
        }
    return 1;
}

// Optimized LU (Version 2.6)
void luFactorization(float** A, float** L, float** U) {
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthr = omp_get_num_threads();

        // First-touch initialization
        #pragma omp for
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                L[i][j] = (i == j) ? 1.0 : 0.0;
                U[i][j] = A[i][j];
            }
        }

        // LU decomposition
        for (int k = 0; k < N; k++) {
            // Step 1: Compute L[i][k] in parallel
            #pragma omp for
            for (int i = k + 1; i < N; i++) {
                L[i][k] = U[i][k] / U[k][k];
            }

            #pragma omp barrier // Ensure L is ready

            // Update U[i][j] (row-major for cache)
            #pragma omp for
            for (int i = k + 1; i < N; i++) {
                for (int j = k; j < N; j++) {
                    U[i][j] -= L[i][k] * U[k][j];
                }
            }

            #pragma omp barrier
        }
    }
}

int main() {
    float** A = allocMatrix();
    float** L = allocMatrix();
    float** U = allocMatrix();

    initMatrix(A);

    double start = omp_get_wtime();
    luFactorization(A, L, U);
    double end = omp_get_wtime();

    printf("Version 4 LU Time: %f seconds\n", end - start);

    /*
    if (verifyLU(A, L, U))
        printf("Verification PASSED\n");
    else
        printf("Verification FAILED\n");
    */

    for (int i = 0; i < N; i++) {
        free(A[i]); free(L[i]); free(U[i]);
    }
    free(A); free(L); free(U);
    return 0;
}
