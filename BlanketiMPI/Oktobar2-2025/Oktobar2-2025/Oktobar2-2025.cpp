#include <iostream>
#include "mpi.h"
#define n 3
#define k 4
#define m 8

int main(int argc, char* argv[])
{
    int rank, size, l;
	int A[n][k], B[k][m];
	MPI_Datatype coltype, colresized;
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	l = m / size;
	int* localB = (int*)malloc(k * l * sizeof(int));
	int* localC = (int*)malloc(n * l * sizeof(int));
	int* C = (int*)malloc(n * m * sizeof(int));
	MPI_Type_vector(k, l, m, MPI_INT, &coltype);
	MPI_Type_commit(&coltype);
	MPI_Type_create_resized(coltype, 0, l * sizeof(int), &colresized);
	MPI_Type_commit(&colresized);

	if (rank == 0)
	{
		for (int i = 0; i < n; i++)
			for (int j = 0; j < k; j++)
				A[i][j] = i + j;
		for (int i = 0; i < k; i++)
			for (int j = 0; j < m; j++)
				B[i][j] = i * j;

		printf("Matrix A:\n");
		fflush(stdout);
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < k; j++)
				printf("%d ", A[i][j]);
			printf("\n");
		}
		printf("Matrix B:\n");
		fflush(stdout);
		for (int i = 0; i < k; i++)
		{
			for (int j = 0; j < m; j++)
				printf("%d ", B[i][j]);
			printf("\n");
		}

		MPI_Bcast(&A[0][0], n * k, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Scatter(&B[0][0], 1, colresized, &localB[0], k * l, MPI_INT, 0, MPI_COMM_WORLD);

		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < l; j++)
			{
				localC[i * l + j] = 0;
				for (int z = 0; z < k; z++)
				{
					localC[i * l + j] += A[i][z] * localB[z * l + j];
				}
			}
		}

		MPI_Gather(&localC[0], n * l, MPI_INT, &C[0], n * l, MPI_INT, 0, MPI_COMM_WORLD);

		printf("Result Matrix C:\n");
		fflush(stdout);
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < m; j++)
				printf("%d ", C[i * m + j]);
			printf("\n");
		}
	}
	else
	{
		MPI_Bcast(&A[0][0], n * k, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Scatter(&B[0][0], 1, colresized, &localB[0], k * l, MPI_INT, 0, MPI_COMM_WORLD);

		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < l; j++)
			{
				localC[i * l + j] = 0;
				for (int z = 0; z < k; z++)
				{
					localC[i * l + j] += A[i][z] * localB[z * l + j];
				}
			}
		}

		MPI_Gather(&localC[0], n * l, MPI_INT, &C[0], n * l, MPI_INT, 0, MPI_COMM_WORLD);
	}

	MPI_Finalize();
    return 0;
}
