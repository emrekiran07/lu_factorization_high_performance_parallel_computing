#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <math.h>

#define N 512
#define B 64
#define EPSILON 1e-1

float** allocMatrix() {
    float** mat = malloc(N * sizeof(float*));
    for (int i = 0; i < N; i++)
        mat[i] = malloc(N * sizeof(float));
    return mat;
}

void copyMatrix(float** src, float** dest) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            dest[i][j] = src[i][j];
}

void initMatrix(float** A) {
    for (int i = 0; i < N; i++) {
        float rowsum = 0.0f;
        for (int j = 0; j < N; j++) {
            A[i][j] = (i == j) ? 0.0f : ((float)rand() / RAND_MAX);
            rowsum += fabsf(A[i][j]);
        }
        A[i][i] = rowsum + 1.0f;
    }
}

int verifyLU(float** A, float** L, float** U) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                float l = (i == k) ? 1.0f : L[i][k];
                sum += l * U[k][j];
            }
            if (fabsf(sum - A[i][j]) > EPSILON * fmaxf(1.0f, fabsf(A[i][j]))) {
                printf("Mismatch at (%d,%d): %.3f != %.3f\n", i, j, sum, A[i][j]);
                return 0;
            }
        }
    }
    return 1;
}

void blockedLU(float** A_orig, float** L, float** U) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            U[i][j] = A_orig[i][j];
            L[i][j] = (i == j) ? 1.0f : 0.0f;
        }

    for (int k = 0; k < N; k += B) {
        int bk = (k + B > N) ? N - k : B; // block size at edge

        // LU on diagonal block
        for (int i = 0; i < bk; i++) {
            int row = k + i;
            for (int j = i + 1; j < bk; j++) {
                int below = k + j;
                float mult = U[below][k + i] / U[row][k + i];
                L[below][k + i] = mult;
                for (int m = k + i; m < k + bk; m++) {
                    U[below][m] -= mult * U[row][m];
                }
            }
        }

        // Update row panel U[k : k+bk][j]
        for (int i = 0; i < bk; i++) {
            int row = k + i;
            for (int j = k + bk; j < N; j++) {
                for (int s = 0; s < i; s++) {
                    U[row][j] -= L[row][k + s] * U[k + s][j];
                }
            }
        }

        // Update column panel L[i][k : k+bk]
        for (int i = k + bk; i < N; i++) {
            for (int j = 0; j < bk; j++) {
                float val = A_orig[i][k + j];  // <<— use original untouched A
                for (int s = 0; s < j; s++) {
                    val -= L[i][k + s] * U[k + s][k + j];
                }
                L[i][k + j] = val / U[k + j][k + j];
            }
        }

        // Update trailing submatrix
        for (int i = k + bk; i < N; i++) {
            for (int j = 0; j < N; j++) {  // Fix: cover full row, not just trailing block
                for (int s = 0; s < bk; s++) {
                    U[i][j] -= L[i][k + s] * U[k + s][j];
                }
            }
        }

    }
}

int main() {
    float** A = allocMatrix();
    float** A_orig = allocMatrix();
    float** L = allocMatrix();
    float** U = allocMatrix();

    initMatrix(A);
    copyMatrix(A, A_orig);

    double start = omp_get_wtime();
    blockedLU(A_orig, L, U);
    double end = omp_get_wtime();

    printf("Version Blocked LU Baseline Time: %f seconds\n", end - start);

    /*
    if (verifyLU(A, L, U))
        printf("Verification PASSED\n");
    else
        printf("Verification FAILED\n");
    */

    for (int i = 0; i < N; i++) {
        free(A[i]); free(A_orig[i]); free(L[i]); free(U[i]);
    }
    free(A); free(A_orig); free(L); free(U);

    return 0;
}
