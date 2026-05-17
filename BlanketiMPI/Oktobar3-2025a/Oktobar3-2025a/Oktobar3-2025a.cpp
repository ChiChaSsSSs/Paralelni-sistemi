#include <iostream>
#include "mpi.h"
#define n 4
#define k 8
#define m 3

int main(int argc, char* argv[])
{
	int rank, size, l;
	int A[n][k], B[k][m];
	MPI_Datatype rowType, rowTypeResized, colType, colTypeResized;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	l = k / size;

	MPI_Type_vector(l * n, 1, size, MPI_INT, &colType);
	MPI_Type_commit(&colType);
	MPI_Type_create_resized(colType, 0, sizeof(int), &colTypeResized);
	MPI_Type_commit(&colTypeResized);

	MPI_Type_vector(l, m, size * m, MPI_INT, &rowType);
	MPI_Type_commit(&rowType);
	MPI_Type_create_resized(rowType, 0, m * sizeof(int), &rowTypeResized);
	MPI_Type_commit(&rowTypeResized);

	int* localA = (int*)malloc(l * n * sizeof(int));
	int* localB = (int*)malloc(l * m * sizeof(int));
	int* localC = (int*)malloc(n * m * sizeof(int));
	int* C = (int*)malloc(n * m * sizeof(int));

	if (rank == 0) {
		for (int i = 0; i < n; i++)
			for (int j = 0; j < k; j++)
				A[i][j] = i + j;

		for (int i = 0; i < k; i++)
			for (int j = 0; j < m; j++)
				B[i][j] = i * j;

		for (int i = 0; i < n * m; i++)
			C[i] = 0;

		printf("Matrix A:\n");
		fflush(stdout);
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < k; j++)
				printf("%d ", A[i][j]);
			printf("\n");
		}
		printf("Matrix B:\n");
		fflush(stdout);
		for (int i = 0; i < k; i++) {
			for (int j = 0; j < m; j++)
				printf("%d ", B[i][j]);
			printf("\n");
		}
	}

	MPI_Scatter(&A[0][0], 1, colTypeResized, &localA[0], l * n, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Scatter(&B[0][0], 1, rowTypeResized, &localB[0], l * m, MPI_INT, 0, MPI_COMM_WORLD);

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			localC[i * m + j] = 0;
			for (int z = 0; z < l; z++)
			{
				localC[i * m + j] += localA[i * l + z] * localB[z * m + j];
			}
		}
	}

	MPI_Reduce(&localC[0], &C[0], n * m, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	if (rank == 0) {
		printf("Result of A * B:\n");
		fflush(stdout);
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++)
				printf("%d ", C[i * m + j]);
			printf("\n");
		}
	}

	free(localA);
	free(localB);
	free(localC);
	free(C);

	MPI_Type_free(&colType);
	MPI_Type_free(&colTypeResized);
	MPI_Type_free(&rowType);
	MPI_Type_free(&rowTypeResized);

	MPI_Finalize();
    return 0;
}
