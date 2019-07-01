#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>

#include "Jacobi.hpp"

using namespace std;

//
// Parallel Order Jabobi
//
int main(int argc, char **argv)
{
	assert(argc > 1);

	// Matrix size
	const int n = atoi(argv[1]);
	assert(n%2==0);

	double *a = new double[n*n];

	// Generate matrix
	Gen_symmat(n,a);

	// debug matrix
	double *oa = new double[n*n];

	#pragma omp parallel
	Copy_symmat(n,a,oa);

	// Eigenvector matrix
	double *v = new double[n*n];
	Set_Iden(n,v);

	double *b = new double[n*n];

	#pragma omp parallel
	Copy_symmat(n,a,b);

	int *top = new int[n/2];
	int *bot = new int[n/2];

	for (int i=0;i<n/2;i++)
	{
		top[i] = i*2;
		bot[i] = i*2+1;
	}

	int k = 0;

	double time = omp_get_wtime();

	while (Off_d(n,a) > EPS)
	{
//		cout << "k = " << k << ", Off(A) = " << Off_d(n,a) << endl;

		for (int t=0; t<n-1; t++)
		{
			#pragma omp parallel
			{
				#pragma omp for
				for (int l=0; l<n/2; l++)
				{
					double c, s;

					int p = (top[l] < bot[l]) ? top[l] : bot[l];
					int q = (top[l] > bot[l]) ? top[l] : bot[l];

					// Generate Givens rotation (c,s)
					Sym_schur2(n,a,p,q,&c,&s);

					// Apply Givens rotation from Left
					GivensL2(n,a,p,q,c,s,b);

					k++;
				}

				#pragma omp for
				for (int l=0; l<n/2; l++)
				{
					double c, s;

					int p = (top[l] < bot[l]) ? top[l] : bot[l];
					int q = (top[l] > bot[l]) ? top[l] : bot[l];

					// Generate Givens rotation (c,s)
					Sym_schur2(n,a,p,q,&c,&s);

					// Apply Givens rotation from Right
					GivensR(n,b,p,q,c,s);

					// Apply Givens rotation from Right
					GivensR(n,v,p,q,c,s);
				}
				// copy a <- b
				Copy_symmat(n,b,a);
			}
			music(n,top,bot);
		}
	}

	time = omp_get_wtime() - time;

	cout << "n = " << n << ", time = " << time << endl;
	cout << "k = " << k << ", Off(A) = " << Off_d(n,a) << endl;

//	cout << "\nA = \n";
//	for (int i=0; i<n; i++) {
//		for (int j=0; j<n; j++)
//			cout << oa[i + j*n] << ", ";
//		cout << endl;
//	}
//	cout << "\nΛ = \n";
//	for (int i=0; i<n; i++) {
//		for (int j=0; j<n; j++)
//			cout << a[i + j*n] << ", ";
//		cout << endl;
//	}
//	cout << "V = \n";
//	for (int i=0; i<n; i++) {
//		for (int j=0; j<n; j++)
//			cout << v[i + j*n] << ", ";
//		cout << endl;
//	}

	cout << "|| V^T A V - Λ|| = " << Residure(n,oa,a,v) << endl;

	delete [] a;
	delete [] v;
	delete [] oa;
	delete [] b;
	delete [] top;
	delete [] bot;

	return EXIT_SUCCESS;
}
