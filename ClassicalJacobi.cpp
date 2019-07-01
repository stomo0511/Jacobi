#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>

#include "Jacobi.hpp"

using namespace std;

//
// Clasiccal Jacobi method
//
int main(int argc, char **argv)
{
	assert(argc > 1);

	// Matrix size
	const int n = atoi(argv[1]);

	double *a = new double[n*n];
	double *v = new double[n*n];

	// Generate matrix
	Gen_symmat(n,a);

	double *oa = new double[n*n];
	Copy_mat(n,a,oa);

	// Eigenvector matrix
	Set_Iden(n,v);

	double c, s;
	int p, q;
	int k = 0;

	double time = omp_get_wtime();

	while (Off_d(n,a) > EPS)
	{
//		cout << "k = " << k << ", Off(A) = " << Off_d(n,a) << endl;

		// Search the maximum element
		Search_max(n,a,&p,&q);
//		cout << "p = " << p << ", q = " << q << endl;

		// Generate Givens rotation (c,s)
		Sym_schur2(n,a,p,q,&c,&s);

		// Apply Givens rotation from Left and Right
		Givens(n,a,p,q,c,s);
//		GivensL(n,a,p,q,c,s);
//		GivensR(n,a,p,q,c,s);

		// Apply Givens rotation from Right
		GivensR(n,v,p,q,c,s);

		k++;
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

	return 0;
}
