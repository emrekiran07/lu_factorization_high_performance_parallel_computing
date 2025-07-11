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

// locked LU
void luFactorization(float** A, float** L, float** U) {
    omp_lock_t locks[N];
    for (int i = 0; i < N; i++) {
        omp_init_lock(&locks[i]);
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthr = omp_get_num_threads();

        // First-touch column init & lock
        for (int col = tid; col < N; col += nthr) {
            for (int i = 0; i < N; i++) {
                L[i][col] = (i == col) ? 1.0 : 0.0;
                U[i][col] = A[i][col];
            }
            omp_set_lock(&locks[col]); // Lock for now, unlock when ready
        }

        #pragma omp barrier // Ensure all init done

        // Elimination and update
        for (int k = 0; k < N; k++) {
            // Thread 0 computes L for column k
            #pragma omp single
            {
                for (int i = k + 1; i < N; i++) {
                    L[i][k] = U[i][k] / U[k][k];
                }
                omp_unset_lock(&locks[k]); // Unlock so others can use L[i][k]
            }

            #pragma omp barrier // Wait until L is ready

            // All threads now update U
            for (int col = tid; col < N; col += nthr) {
                // Wait if column k is still locked
                omp_set_lock(&locks[k]); // Ensure safe read of L[i][k]
                omp_unset_lock(&locks[k]);

                for (int i = k + 1; i < N; i++) {
                    U[i][col] -= L[i][k] * U[k][col];
                }
            }

            #pragma omp barrier // Sync before next k
        }
    }

    // Cleanup locks
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

    printf("Version 3 LU Time: %f seconds\n", end - start);

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
