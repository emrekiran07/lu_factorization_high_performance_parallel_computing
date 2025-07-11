#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define N 1000
#define EPSILON 1e-2

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

void luFactorization(float** A, float** L, float** U) {
    omp_lock_t locks[N];
    for (int i = 0; i < N; i++) {
        omp_init_lock(&locks[i]);
    }

    int k;

    #pragma omp parallel private(k)
    {
        int thrid = omp_get_thread_num();
        int nthr = omp_get_num_threads();

        // First touch and column lock setup
        for (int col = thrid; col < N; col += nthr) {
            for (int i = 0; i < N; i++) {
                L[i][col] = (i == col) ? 1.0 : 0.0;
                U[i][col] = A[i][col];
            }
            omp_set_lock(&locks[col]);  // lock all initially
        }

        #pragma omp barrier

        // Special case for k = 0
        if (thrid == 0) {
            for (int i = 1; i < N; i++) {
                L[i][0] = U[i][0] / U[0][0];
            }
            omp_unset_lock(&locks[0]);
        }

        // Main loop
        for (k = 1; k < N; k++) {
            omp_set_lock(&locks[k]);  // wait until safe to compute L[i][k]

            if (thrid == 0) {
                for (int i = k + 1; i < N; i++) {
                    L[i][k] = U[i][k] / U[k][k];
                }
                omp_unset_lock(&locks[k]);  // allow others to read L[i][k]
            }

            #pragma omp barrier

            // Update U in parallel
            int start = (k / nthr) * nthr;
            if (start + thrid < k + 1) start += nthr;

            for (int col = start + thrid; col < N; col += nthr) {
                for (int i = k + 1; i < N; i++) {
                    U[i][col] -= L[i][k] * U[k][col];
                }

                // precompute L for next iteration
                if (col == k + 1 && k + 1 < N) {
                    omp_unset_lock(&locks[k + 1]);  // allow L[k+1] next time
                }
            }

            #pragma omp barrier
        }
    }

    for (int i = 0; i < N; i++) {
        omp_destroy_lock(&locks[i]);
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

    printf("Version 5 LU Time: %f seconds\n", end - start);

    
    if (verifyLU(A, L, U))
        printf("Verification PASSED\n");
    else
        printf("Verification FAILED\n");
    

    for (int i = 0; i < N; i++) {
        free(A[i]); free(L[i]); free(U[i]);
    }
    free(A); free(L); free(U);
    return 0;
}
