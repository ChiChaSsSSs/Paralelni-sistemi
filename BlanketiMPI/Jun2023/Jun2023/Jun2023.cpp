#include <iostream>
#include "mpi.h"
#define N 6

typedef struct {
	int id;
	char ime[20];
	char prezime[20];
	float plata;
} Zaposleni;


int main(int argc, char* argv[])
{
	int rank, size, l;
	int idMin;
	float plataMin;
	struct {
		float plataMin;
		int idMin;
	}ulaz, izlaz;

	MPI_Datatype ZaposleniType;
	int blockLength[4] = { 1,20,20,1 };
	MPI_Datatype types[4] = { MPI_INT, MPI_CHAR, MPI_CHAR, MPI_FLOAT };
	MPI_Aint displacements[4];

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	l = N / size;

	Zaposleni* sviZaposleni = (Zaposleni*)malloc(N * sizeof(Zaposleni));
	Zaposleni* grupaZaposlenih = (Zaposleni*)malloc(l * sizeof(Zaposleni));

	MPI_Get_address(&(sviZaposleni[0].id), &displacements[0]);
	MPI_Get_address(&(sviZaposleni[0].ime), &displacements[1]);
	MPI_Get_address(&(sviZaposleni[0].prezime), &displacements[2]);
	MPI_Get_address(&(sviZaposleni[0].plata), &displacements[3]);
	displacements[3] -= displacements[0];
	displacements[2] -= displacements[0];
	displacements[1] -= displacements[0];
	displacements[0] = 0;
	MPI_Type_create_struct(4, blockLength, displacements, types, &ZaposleniType);
	MPI_Type_commit(&ZaposleniType);

	if (rank == 0)
	{
		sviZaposleni[0] = { 1, "Pera", "Peric", 50000 };
		sviZaposleni[1] = { 2, "Mika", "Mikic", 60000 };
		sviZaposleni[2] = { 3, "Zika", "Zikic", 55000 };
		sviZaposleni[3] = { 4, "Laza", "Lazic", 70000 };
		sviZaposleni[4] = { 5, "Sima", "Simic", 65000 };
		sviZaposleni[5] = { 6, "Dule", "Dulic", 72000 };
	}

	MPI_Scatter(sviZaposleni, l, ZaposleniType, grupaZaposlenih, l, ZaposleniType, 0, MPI_COMM_WORLD);

	idMin = grupaZaposlenih[0].id;
	plataMin = grupaZaposlenih[0].plata;
	for (int i = 1; i < l; i++)
	{
		if (grupaZaposlenih[i].plata < plataMin)
		{
			plataMin = grupaZaposlenih[i].plata;
			idMin = grupaZaposlenih[i].id;
		}
	}

	ulaz.plataMin = plataMin;
	ulaz.idMin = idMin;
	
	MPI_Reduce(&ulaz, &izlaz, 1, MPI_FLOAT_INT, MPI_MINLOC, 0, MPI_COMM_WORLD);

	if (rank == 0)
	{
		printf("Zaposleni sa najmanjom platom ima ID: %d i platu: %.2f\n", izlaz.idMin, izlaz.plataMin);
	}

	MPI_Finalize();	
    return 0;
}
