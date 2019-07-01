#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>

#include "Jacobi.hpp"

using namespace std;

//
// Cyclic-by-Row Jabobi
//
int main(int argc, char **argv)
{
	assert(argc > 1);

	// Matrix size
//	const int n = 4;
	const int n = atoi(argv[1]);
//	cout << "n = " << n << endl;

	double *a = new double[n*n];
	double *v = new double[n*n];

	// Generate matrix
	Gen_symmat(n,a);

	double *oa = new double[n*n];
	Copy_mat(n,a,oa);

	// Eigenvector matrix
	Set_Iden(n,v);

//	cout << "A = \n";
//	for (int i=0; i<n; i++) {
//		for (int j=0; j<n; j++)
//			cout << a[i + j*n] << ", ";
//		cout << endl;
//	}
//	cout << "Off(A) = " << ofd << endl;
//	cout << "V = \n";
//	for (int i=0; i<n; i++) {
//		for (int j=0; j<n; j++)
//			cout << v[i + j*n] << ", ";
//		cout << endl;
//	}

	double c, s;
	int k = 0;

	double time = omp_get_wtime();

	while (Off_d(n,a) > EPS)
	{
//		cout << "\nk = " << k << ", Off(A) = " << ofd << endl;

		for (int i=0; i<n; i++)
			for (int j=0; j<i; j++) {
//				cout << "p = " << i << ", q = " << j << endl;

				// Generate Givens rotation (c,s)
				Sym_schur2(n,a,i,j,&c,&s);
//				cout << "c = " << c << ", s = " << s << endl;

				// Apply Givens rotation
				Givens(n,a,i,j,c,s);

				// Apply Givens rotation from Right
				GivensR(n,v,i,j,c,s);

				k++;
			}
	}

	time = omp_get_wtime() - time;

	cout << "n = " << n << ", time = " << time << endl;
	cout << "k = " << k << ", Off(A) = " << Off_d(n,a) << endl;


//	cout << "\nA = \n";
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
	delete [] oa;
	delete [] v;

	return 0;
}
