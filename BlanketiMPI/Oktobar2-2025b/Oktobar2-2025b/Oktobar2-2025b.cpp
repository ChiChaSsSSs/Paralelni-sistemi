#include <iostream>
#include <cmath>
#include "mpi.h"
#define M 12

int main(int argc, char* argv[])
{
	int rank, size, row, column, n;
    MPI_Datatype triangleMatrix;
	MPI_Status status;
	MPI_Group matrixGroup, worldGroup;
	MPI_Comm matrixComm;

	MPI_Init(&argc, &argv); 
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

    n = sqrt(size);
    row = rank / n;
	column = rank % n;
	
	if (row > column)
	{
		typedef struct {
			char jmbg[14];
			char ime[15];
			char prezime[20];
			double plata;
		} Zaposleni;

		MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);

		int newCommSize = n * (n - 1) / 2;
		int newRank;
		Zaposleni zaposleni[M];
		Zaposleni* grupaZaposlenih = (Zaposleni*)malloc(M / newCommSize * sizeof(Zaposleni));

		int* ranks = (int*)malloc(newCommSize * sizeof(int));

		int index = 0;
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < i; j++)
			{
				ranks[index++] = i * n + j;
			}
		}

		MPI_Group_incl(worldGroup, newCommSize, ranks, &matrixGroup);
		MPI_Comm_create(MPI_COMM_WORLD, matrixGroup, &matrixComm);	

		MPI_Comm_rank(matrixComm, &newRank);

		if (newRank == 0)
		{
			for (int i = 0; i < M; i++)
			{
				sprintf(zaposleni[i].jmbg, "123456789012%d", i);
				sprintf(zaposleni[i].ime, "Ime%d", i);
				sprintf(zaposleni[i].prezime, "Prezime%d", i);
				zaposleni[i].plata = 1000.0 + i * 100.0;
			}
		}

		MPI_Scatter(zaposleni, M / newCommSize, MPI_BYTE, grupaZaposlenih, M / newCommSize, MPI_BYTE, 0, matrixComm);

		for (int i = 0; i < M / newCommSize; i++)
		{
			std::cout << "Proces " << rank << " je dobio zaposlenog: "
				<< grupaZaposlenih[i].jmbg << ", "
				<< grupaZaposlenih[i].ime << ", "
				<< grupaZaposlenih[i].prezime << ", "
				<< grupaZaposlenih[i].plata << std::endl;
		}
	}
	else
	{
		matrixComm = MPI_COMM_NULL;
	}

    MPI_Finalize();
    return 0;
}

