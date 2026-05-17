#include <iostream>
#include "mpi.h"
#define N 6

int main(int argc, char* argv[])
{
	int rank, size, row, col;
	int A[N][N], B[N][N], C[N][N];
	int local_A[N], local_B[N], local_C;
	MPI_Datatype colType, colTypeResized;
	MPI_Comm colComm, rowComm;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	row = rank / N;
	col = rank % N;

	MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm);
	MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm);

	MPI_Type_vector(N, 1, N, MPI_INT, &colType);
	MPI_Type_commit(&colType);
	MPI_Type_create_resized(colType, 0, sizeof(int), &colTypeResized);
	MPI_Type_commit(&colTypeResized);

	if (rank == 0)
	{
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				A[i][j] = i + j;
				B[i][j] = i * j;
			}
		}

		printf("Matrix A:\n");
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				printf("%d ", A[i][j]);
			}
			printf("\n");
		}
		printf("Matrix B:\n");
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				printf("%d ", B[i][j]);
			}
			printf("\n");
		}
	}

	if (col == 0) 
	{
		MPI_Scatter(&A[0][0], N, MPI_INT, &local_A[0], N, MPI_INT, 0, colComm);
	}
	MPI_Bcast(&local_A[0], N, MPI_INT, 0, rowComm);

	if (row == 0)
	{
		MPI_Scatter(&B[0][0], 1, colTypeResized, &local_B[0], N, MPI_INT, 0, rowComm);
	}
	MPI_Bcast(&local_B[0], N, MPI_INT, 0, colComm);

	local_C = 0;
	for (int i = 0; i < N; i++)
	{
		local_C += local_A[i] * local_B[i];
	}

	MPI_Gather(&local_C, 1, MPI_INT, &C[0][0], 1, MPI_INT, 0, MPI_COMM_WORLD);

	if (rank == 0)
	{
		printf("Result Matrix C:\n");
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				printf("%d ", C[i][j]);
			}
			printf("\n");
		}
	}

	MPI_Finalize();
	return 0;
}
