#include <iostream>
#include <cmath>
#include "mpi.h"

int main(int argc, char* argv[])
{
	int rank, size, n, row, column;
	MPI_Group worldGroup, diagonalGroup;
	MPI_Comm diagonalComm;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	n = sqrt(size);
	row = rank / n;
	column = rank % n;

	MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);

	int* ranks = (int*)malloc(n * sizeof(int));
	for (int i = 0; i < n; i++)
		ranks[i] = i * n + i;

	MPI_Group_incl(worldGroup, n, ranks, &diagonalGroup);
	MPI_Comm_create(MPI_COMM_WORLD, diagonalGroup, &diagonalComm);

	if (row == column)
	{
		int newRank;
		int* matrix = (int*)malloc(n * n * sizeof(int));
		int* columnData = (int*)malloc(n * sizeof(int));
		MPI_Datatype columnType, columnTypeResized;

		MPI_Comm_rank(diagonalComm, &newRank);

		MPI_Type_vector(n, 1, n, MPI_INT, &columnType);
		MPI_Type_commit(&columnType);
		MPI_Type_create_resized(columnType, 0, sizeof(int), &columnTypeResized);
		MPI_Type_commit(&columnTypeResized);

		if (newRank == 0)
		{
			for (int i = 0; i < n * n; i++)
				matrix[i] = i;
			printf("Matrix:\n");
			fflush(stdout);
			for (int i = 0; i < n; i++)
			{
				for (int j = 0; j < n; j++)
					printf("%d ", matrix[i * n + j]);
				printf("\n");
				fflush(stdout);
			}
		}

		MPI_Scatter(&matrix[0], 1, columnTypeResized, &columnData[0], n, MPI_INT, 0, diagonalComm);

		printf("Process %d received column: ", rank);
		for (int i = 0; i < n; i++)
			printf("%d ", columnData[i]);
	}

	MPI_Finalize();
	return 0;
}