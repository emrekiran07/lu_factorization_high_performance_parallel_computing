#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define N 5000  // matrix size

// Allocate and initialize matrices
float** allocMatrix() {
    float** mat = (float**)malloc(N * sizeof(float*));
    for (int i = 0; i < N; i++)
        mat[i] = (float*)malloc(N * sizeof(float));
    return mat;
}

void initMatrix(float** A) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = (i + j) % 10 + 1;
}

// LU decomposition with OpenMP (unsafe baseline)
void luFactorization(float** A, float** L, float** U) {
    // Initialize L as identity, U as copy of A
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            L[i][j] = (i == j) ? 1.0 : 0.0;
            U[i][j] = A[i][j];
        }

    #pragma omp parallel
    {
        for (int k = 0; k < N; k++) {
            // Compute multipliers
            #pragma omp for
            for (int i = k + 1; i < N; i++) {
                L[i][k] = U[i][k] / U[k][k];
            }

            // Update U
            #pragma omp for
            for (int i = k + 1; i < N; i++) {
                for (int j = k; j < N; j++) {
                    U[i][j] -= L[i][k] * U[k][j];
                }
            }
        }
    }
}

int verifyLU(float** A, float** L, float** U) {
    float EPSILON = 1e-2;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += L[i][k] * U[k][j];
            }
            if (fabs(sum - A[i][j]) > EPSILON) {
                printf("Mismatch at A[%d][%d]: expected %.2f, got %.2f\n", i, j, A[i][j], sum);
                return 0; // Fail
            }
        }
    }
    return 1; // Pass
}

int main() {
    float** A = allocMatrix();
    float** L = allocMatrix();
    float** U = allocMatrix();

    initMatrix(A);

    double start = omp_get_wtime();
    luFactorization(A, L, U);
    double end = omp_get_wtime();

    printf("Baseline LU Time: %f seconds\n", end - start);
    
    /* if (verifyLU(A, L, U))
        printf("Verification PASSED\n");
    else
        printf("Verification FAILED\n");
    */

    // Cleanup
    for (int i = 0; i < N; i++) {
        free(A[i]); free(L[i]); free(U[i]);
    }
    free(A); free(L); free(U);

    return 0;
}

