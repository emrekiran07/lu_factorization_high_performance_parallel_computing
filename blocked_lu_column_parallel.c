#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 512
#define B 64
#define EPSILON (N > 512 ? 1e-1f : 1e-2f)

// 1D column-major allocation
float* allocMatrix() {
    return (float*) malloc(N * N * sizeof(float));
}

#define IDX(i, j) ((j) * N + (i))  // column-major access

void initMatrix(float* A) {
    for (int j = 0; j < N; j++) {
        float colsum = 0.0f;
        for (int i = 0; i < N; i++) {
            if (i != j) {
                A[IDX(i,j)] = (float)rand() / RAND_MAX;
                colsum += fabsf(A[IDX(i,j)]);
            }
        }
        A[IDX(j,j)] = colsum + 1.0f;
    }
}

void copyMatrix(float* src, float* dst) {
    for (int i = 0; i < N*N; i++)
        dst[i] = src[i];
}

void multiplyLU(float* L, float* U, float* LU) {
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                float l = (i == k) ? 1.0f : L[IDX(i,k)];
                float u = U[IDX(k,j)];
                sum += l * u;
            }
            LU[IDX(i,j)] = sum;
        }
    }
}

int verifyLU(float* A, float* L, float* U) {
    float* LU = allocMatrix();
    multiplyLU(L, U, LU);

    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            float diff = fabsf(LU[IDX(i,j)] - A[IDX(i,j)]);
            float tol = EPSILON * fmaxf(1.0f, fabsf(A[IDX(i,j)]));
            if (diff > tol) {
                printf("Mismatch at (%d,%d): %.3f != %.3f\n", i, j, LU[IDX(i,j)], A[IDX(i,j)]);
                free(LU);
                return 0;
            }
        }
    }
    free(LU);
    return 1;
}

void blockedLU(float* A, float* L, float* U) {
    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++) {
            L[IDX(i,j)] = (i == j) ? 1.0f : 0.0f;
            U[IDX(i,j)] = A[IDX(i,j)];
        }

    for (int k = 0; k < N; k += B) {
        int bk = (k + B > N) ? (N - k) : B;

        // LU factorization on diagonal block
        for (int j = 0; j < bk; j++) {
            int col = k + j;
            for (int i = col + 1; i < k + bk; i++) {
                float mult = U[IDX(i,col)] / U[IDX(col,col)];
                L[IDX(i,col)] = mult;
                for (int jj = col; jj < k + bk; jj++) {
                    U[IDX(i,jj)] -= mult * U[IDX(col,jj)];
                }
            }
        }

        // Update U panel (rows)
        #pragma omp parallel for collapse(2)
        for (int i = k; i < k + bk; i++) {
            for (int j = k + bk; j < N; j++) {
                float sum = 0.0f;
                for (int s = k; s < i; s++) {
                    sum += L[IDX(i,s)] * U[IDX(s,j)];
                }
                U[IDX(i,j)] -= sum;
            }
        }

        // Update L panel (columns)
        #pragma omp parallel for collapse(2)
        for (int i = k + bk; i < N; i++) {
            for (int j = k; j < k + bk; j++) {
                float sum = 0.0f;
                for (int s = k; s < j; s++) {
                    sum += L[IDX(i,s)] * U[IDX(s,j)];
                }
                L[IDX(i,j)] = (U[IDX(j,j)] == 0.0f) ? 0.0f : (U[IDX(i,j)] - sum) / U[IDX(j,j)];
                U[IDX(i,j)] = 0.0f; 
            }
        }

        // Update trailing submatrix
        #pragma omp parallel for collapse(2)
        for (int i = k + bk; i < N; i++) {
            for (int j = k + bk; j < N; j++) {
                float sum = 0.0f;
                for (int s = k; s < k + bk; s++) {
                    sum += L[IDX(i,s)] * U[IDX(s,j)];
                }
                U[IDX(i,j)] -= sum;
            }
        }
    }
}

int main() {
    srand(42);

    float* A = allocMatrix();
    float* A_orig = allocMatrix();
    float* L = allocMatrix();
    float* U = allocMatrix();

    initMatrix(A);
    copyMatrix(A, A_orig);

    double start = omp_get_wtime();
    blockedLU(A_orig, L, U);
    double end = omp_get_wtime();

    printf("Blocked LU Column-Major Time: %f seconds\n", end - start);

    /*
    if (verifyLU(A, L, U))
        printf("Verification PASSED\n");
    else
        printf("Verification FAILED\n");
    */

    free(A); free(A_orig); free(L); free(U);
    return 0;
}
