#include <iostream>
#include "mpi.h"
#define m 4
#define n 6

int main(int argc, char* argv[])
{
    int rank, size, count;
    int A[m][n], B[n], C[m];
	MPI_Datatype column_type, resized_column_type;
	MPI_Datatype vector_type, resized_vector_type;

    MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

    count = n / size;
    int* local_A = (int*)malloc(m * count * sizeof(int));
	int* local_B = (int*)malloc(count * sizeof(int));
	int* local_C = (int*)malloc(m * sizeof(int));

	MPI_Type_vector(m * count, 1, size, MPI_INT, &column_type);
	MPI_Type_commit(&column_type);
	MPI_Type_create_resized(column_type, 0, sizeof(int), &resized_column_type);
	MPI_Type_commit(&resized_column_type);

	MPI_Type_vector(count, 1, size, MPI_INT, &vector_type);
	MPI_Type_commit(&vector_type);
	MPI_Type_create_resized(vector_type, 0, sizeof(int), &resized_vector_type);
	MPI_Type_commit(&resized_vector_type);

	if (rank == 0)
	{
		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++)
				A[i][j] = i + j;
		for (int i = 0; i < n; i++)
			B[i] = i;

		printf("Matrix A:\n");
		fflush(stdout);
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
				printf("%d ", A[i][j]);
			printf("\n");
		}
		printf("Vector B:\n");
		fflush(stdout);
		for (int i = 0; i < n; i++)
			printf("%d ", B[i]);
		printf("\n");
	}

	MPI_Scatter(&A[0][0], 1, resized_column_type, &local_A[0], m * count, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Scatter(&B[0], 1, resized_vector_type, &local_B[0], count, MPI_INT, 0, MPI_COMM_WORLD);

	for (int i = 0; i < m; i++)
	{
		local_C[i] = 0;
		for (int j = 0; j < count; j++)
		{
			local_C[i] += local_A[i * count + j] * local_B[j];
		}
	}

	MPI_Reduce(local_C, C, m, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	if (rank == 0)
	{
		printf("Result Vector C:\n");
		fflush(stdout);
		for (int i = 0; i < m; i++)
			printf("%d ", C[i]);
		printf("\n");
	}

    MPI_Finalize();
    return 0;
}
