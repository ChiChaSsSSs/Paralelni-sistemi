#include <iostream>
#include <cmath>
#include "mpi.h"
#define N 20

typedef struct {
	int brojIndeksa;
	char ime[20];
	char prezime[20];
	float prosek;
} Student;

int main(int argc, char* argv[])
{
	int rank, size, row, column, n;
	MPI_Datatype studentType;
	MPI_Group worldGroup, subGroup;
	MPI_Comm subComm;
	Student* sviStudenti = (Student*)malloc(N * sizeof(Student));

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	n = sqrt(size);
	row = rank / n;
	column = rank % n;

	MPI_Datatype types[4] = { MPI_INT, MPI_CHAR, MPI_CHAR, MPI_FLOAT };
	int blockLenghts[4] = { 1, 20, 20, 1 };
	MPI_Aint offsets[4];
	MPI_Get_address(&(sviStudenti[0].brojIndeksa), &offsets[0]);
	MPI_Get_address(&(sviStudenti[0].ime), &offsets[1]);
	MPI_Get_address(&(sviStudenti[0].prezime), &offsets[2]);
	MPI_Get_address(&(sviStudenti[0].prosek), &offsets[3]);
	offsets[3] = offsets[3] - offsets[0];
	offsets[2] = offsets[2] - offsets[0];
	offsets[1] = offsets[1] - offsets[0];
	offsets[0] = 0;
	MPI_Type_create_struct(4, blockLenghts, offsets, types, &studentType);
	MPI_Type_commit(&studentType);

	if (rank == 0)
	{
		for (int i = 0; i < N; i++)
		{
			sviStudenti[i].brojIndeksa = 1000 + i;
			sprintf(sviStudenti[i].ime, "Ime%d", i);
			sprintf(sviStudenti[i].prezime, "Prezime%d", i);
			sviStudenti[i].prosek = 6.0 + (float)(i % 5);
		}
	}

	MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);
	int* ranks = (int*)malloc(n * sizeof(int));

	int index = 0;
	for (int i = 0; i < n; i++)
		ranks[index++] = i * n + i;

	MPI_Group_incl(worldGroup, n, ranks, &subGroup);
	MPI_Comm_create(MPI_COMM_WORLD, subGroup, &subComm);

	if (row == column)
	{
		int newRank;
		MPI_Group_rank(subGroup, &newRank);

		Student* grupaStudenata = (Student*)malloc(N / n * sizeof(Student));

		MPI_Scatter(sviStudenti, N / n, studentType, grupaStudenata, N / n, studentType, 0, subComm);

		for (int i = 0; i < N / n; i++)
		{
			printf("Proces %d:", newRank);
			printf(" %d %s %s %.2f\n", grupaStudenata[i].brojIndeksa, grupaStudenata[i].ime,
				grupaStudenata[i].prezime, grupaStudenata[i].prosek);
		}
	}

	MPI_Finalize();
    return 0;
}
