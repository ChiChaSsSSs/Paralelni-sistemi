#include <iostream>
#include "mpi.h"
#define n 4
#define s 3
#define k 8

int main(int argc, char* argv[])
{
    int rank, size, l;
    int A[n][s], B[s][k];
	MPI_Datatype col_type, col_type_resized;
	MPI_Datatype res_col_type, res_col_type_resized;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	l = k / size;

	int* local_B = (int*)malloc(l * s * sizeof(int));
	int* local_C = (int*)malloc(l * n * sizeof(int));
	int* C = (int*)malloc(n * k * sizeof(int));

	MPI_Type_vector(s * l, 1, size, MPI_INT, &col_type);
	MPI_Type_commit(&col_type);
	MPI_Type_create_resized(col_type, 0, sizeof(int), &col_type_resized);
	MPI_Type_commit(&col_type_resized);

	MPI_Type_vector(n * l, 1, size, MPI_INT, &res_col_type);
	MPI_Type_commit(&res_col_type);
	MPI_Type_create_resized(res_col_type, 0, sizeof(int), &res_col_type_resized);
	MPI_Type_commit(&res_col_type_resized);

	if (rank == 0)
	{
		for (int i = 0; i < n; i++)
			for (int j = 0; j < s; j++)
				A[i][j] = i + j;
		for (int i = 0; i < s; i++)
			for (int j = 0; j < k; j++)
				B[i][j] = i * j;

		printf("Matrix A:\n");
		fflush(stdout);
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < s; j++)
				printf("%d ", A[i][j]);
			printf("\n");
		}
		printf("Matrix B:\n");
		fflush(stdout);
		for (int i = 0; i < s; i++)
		{
			for (int j = 0; j < k; j++)
				printf("%d ", B[i][j]);
			printf("\n");
		}

		MPI_Bcast(&A[0][0], n * s, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Scatter(&B[0][0], 1, col_type_resized, local_B, l * s, MPI_INT, 0, MPI_COMM_WORLD);

		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < l; j++)
			{
				local_C[i * l + j] = 0;
				for (int z = 0; z < s; z++)
				{
					local_C[i * l + j] += A[i][z] * local_B[z * l + j];
				}
			}
		}

		MPI_Gather(&local_C[0], l * n, MPI_INT, &C[0], 1, res_col_type_resized, 0, MPI_COMM_WORLD);

		printf("Result of A x B:\n");
		fflush(stdout);
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < k; j++)
				printf("%d ", C[i * k + j]);
			printf("\n");
		}
	}
	else
	{
		MPI_Bcast(&A[0][0], n * s, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Scatter(&B[0][0], 1, col_type_resized, local_B, l * s, MPI_INT, 0, MPI_COMM_WORLD);

		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < l; j++)
			{
				local_C[i * l + j] = 0;
				for (int z = 0; z < s; z++)
				{
					local_C[i * l + j] += A[i][z] * local_B[z * l + j];
				}
			}
		}

		MPI_Gather(&local_C[0], l * n, MPI_INT, &C[0], 1, res_col_type_resized, 0, MPI_COMM_WORLD);
	}

	MPI_Finalize();
    return 0;
}
