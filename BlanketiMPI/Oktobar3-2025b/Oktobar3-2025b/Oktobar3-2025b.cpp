#include <iostream>
#include <cmath>
#include "mpi.h"

int main(int argc, char* argv[])
{
	int rank, size, row, column, n, newSize;
	MPI_Group worldGroup, triangleGroup;
	MPI_Comm trinagleComm;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	n = sqrt(size);
	row = rank / n;
	column = rank % n;
	newSize = n * (n - 1) / 2;

	MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);

	int* ranks = (int*)malloc(newSize * sizeof(int));
	int index = 0;
	for (int i = 1; i < n; i++)
	{
		for (int j = 0; j < i; j++)
			ranks[index++] = i * n + j;
	}

	MPI_Group_incl(worldGroup, newSize, ranks, &triangleGroup);
	MPI_Comm_create(MPI_COMM_WORLD, triangleGroup, &trinagleComm);

	if (row > column)
	{
		int newRank;
		int* matrix = (int*)malloc(n * n * sizeof(int));
		int* diagonal = (int*)malloc(n * sizeof(int));
		MPI_Datatype diagonalType;

		MPI_Comm_rank(trinagleComm, &newRank);

		MPI_Type_vector(n, 1, n + 1, MPI_INT, &diagonalType);
		MPI_Type_commit(&diagonalType);

		if (newRank == 0)
		{
			for (int i = 0; i < n * n; i++)
				matrix[i] = i + 1;

			printf("Matrix:\n");
			fflush(stdout);
			for (int i = 0; i < n; i++)
			{
				for (int j = 0; j < n; j++)
				{
					printf("%d ", matrix[i * n + j]);
					fflush(stdout);
				}
				printf("\n");
				fflush(stdout);
			}

			MPI_Bcast(&matrix[0], 1, diagonalType, 0, trinagleComm);
		}
		else
		{
			MPI_Bcast(&diagonal[0], n, MPI_INT, 0, trinagleComm);

			printf("Process %d received diagonal: ", rank);
			fflush(stdout);
			for (int i = 0; i < n; i++)
			{
				printf("%d ", diagonal[i]);
			}
		}
	}

	MPI_Finalize();
    return 0;
}
