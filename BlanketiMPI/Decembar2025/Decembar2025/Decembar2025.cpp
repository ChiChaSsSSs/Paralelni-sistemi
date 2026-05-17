#include <iostream>
#include "mpi.h"
#define r 8
#define m 4
#define k 6

int main(int argc, char* argv[])
{
    int rank, size, l;
    int A[r][m], B[m][k];

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	l = r / size;
	int* local_A = (int*)malloc(l * m * sizeof(int));
	int* local_C = (int*)malloc(l * k * sizeof(int));
	int* C = (int*)malloc(r * k * sizeof(int));

	if (rank == 0) 
	{
		for (int i = 0; i < r; i++)
			for (int j = 0; j < m; j++)
				A[i][j] = i + j;
		for (int i = 0; i < m; i++)
			for (int j = 0; j < k; j++)
				B[i][j] = i * j;

		printf("Matrix A:\n");
		fflush(stdout);
		for (int i = 0; i < r; i++)
		{
			for (int j = 0; j < m; j++)
				printf("%d ", A[i][j]);
			printf("\n");
		}
		printf("Matrix B:\n");
		fflush(stdout);
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < k; j++)
				printf("%d ", B[i][j]);
			printf("\n");
		}

		MPI_Scatter(&A[0][0], l * m, MPI_INT, &local_A[0], l * m, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&B[0][0], m * k, MPI_INT, 0, MPI_COMM_WORLD);

		for (int i = 0; i < l; i++)
		{
			for (int j = 0; j < k; j++) 
			{
				local_C[i * k + j] = 0;
				for (int z =0;z<m;z++) 
				{
					local_C[i * k + j] += local_A[i * m + z] * B[z][j];
				}
			}
		}

		MPI_Gather(&local_C[0], l * k, MPI_INT, &C[0], l * k, MPI_INT, 0, MPI_COMM_WORLD);

		printf("Result of A x B:\n");
		fflush(stdout);
		for (int i = 0; i < r; i++)
		{
			for (int j = 0; j < k; j++)
				printf("%d ", C[i * k + j]);
			printf("\n");
		}
	}
	else 
	{
		MPI_Scatter(&A[0][0], l * m, MPI_INT, &local_A[0], l * m, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&B[0][0], m * k, MPI_INT, 0, MPI_COMM_WORLD);

		for (int i = 0; i < l; i++)
		{
			for (int j = 0; j < k; j++)
			{
				local_C[i * k + j] = 0;
				for (int z = 0; z < m; z++)
				{
					local_C[i * k + j] += local_A[i * m + z] * B[z][j];
				}
			}
		}

		MPI_Gather(&local_C[0], l * k, MPI_INT, &C[0], l * k, MPI_INT, 0, MPI_COMM_WORLD);
	}

	MPI_Finalize();
    return 0;
}
