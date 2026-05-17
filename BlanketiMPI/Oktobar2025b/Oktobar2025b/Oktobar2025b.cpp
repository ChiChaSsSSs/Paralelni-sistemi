#include <iostream>
#include <string>
#include <cmath>
#include "mpi.h"

typedef struct {
	char jmbg[14];
	char ime[20];
	char prezime[20];
	float plata;
} Zaposleni;

int main(int argc, char* argv[])
{
	int rank, size, row, column, n, groupSize;
	MPI_Group worldGroup, triangleGroup;
	MPI_Comm triangleComm;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	n = (int)sqrt(size);
	row = rank / n;
	column = rank % n;

	MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);

	groupSize = (n * (n + 1)) / 2;
	int* ranks = (int*)malloc(groupSize * sizeof(int));
	int index = 0;
	for (int i = 0; i < n; i++)
	{
		for (int j = i; j < n; j++)
		{
			ranks[index] = i * n + j;
			index++;
		}
	}

	MPI_Group_incl(worldGroup, groupSize, ranks, &triangleGroup);
	MPI_Comm_create(MPI_COMM_WORLD, triangleGroup, &triangleComm);

	if (column >= row)
	{
		int newRank;
		Zaposleni* zaposleni = (Zaposleni*)malloc(groupSize * sizeof(Zaposleni));
		Zaposleni primljeni;
		MPI_Comm_rank(triangleComm, &newRank);

		MPI_Datatype zaposleniType;
		MPI_Datatype types[4] = { MPI_CHAR,MPI_CHAR,MPI_CHAR, MPI_FLOAT };
		int blockLengths[4] = { 14,20,20,1 };
		MPI_Aint offsets[4];
		MPI_Get_address(&(zaposleni[0].jmbg), &offsets[0]);
		MPI_Get_address(&(zaposleni[0].ime), &offsets[1]);
		MPI_Get_address(&(zaposleni[0].prezime), &offsets[2]);
		MPI_Get_address(&(zaposleni[0].plata), &offsets[3]);
		offsets[3] = offsets[3] - offsets[0];
		offsets[2] = offsets[2] - offsets[0];
		offsets[1] = offsets[1] - offsets[0];
		offsets[0] = 0;
		MPI_Type_create_struct(4, blockLengths, offsets, types, &zaposleniType);
		MPI_Type_commit(&zaposleniType);

		if (newRank == 0)
		{
			for (int i = 0; i < groupSize; i++)
			{
				strcpy(zaposleni[i].jmbg, "1234567890123");
				strcpy(zaposleni[i].ime, "Ime");
				strcpy(zaposleni[i].prezime, "Prezime");
				zaposleni[i].plata = 1000.0f + i;
			}
		}

		MPI_Scatter(&zaposleni[0], 1, zaposleniType, &primljeni, 1, zaposleniType, 0, triangleComm);

		printf("Proces %d primio: JMBG: %s, Ime: %s, Prezime: %s, Plata: %.2f\n", rank, primljeni.jmbg, primljeni.ime, primljeni.prezime, primljeni.plata);
	}

	MPI_Finalize();
    return 0;
}
