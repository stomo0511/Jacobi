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
//	int n = 4;
	const int n = atoi(argv[1]);
	assert(n%2==0);

//	cout << "n = " << n << endl;

	double *a = new double[n*n];
	double *v = new double[n*n];

	// Generate matrix
	Gen_mat(n,a);

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

	int *top = new int[n/2];
	int *bot = new int[n/2];

	for (int i=0;i<n/2;i++)
	{
		top[i] = i*2;
		bot[i] = i*2+1;
	}

	double c, s;
	int p, q;
	int k = 0;

	double time = omp_get_wtime();

	while (Off_d(n,a) > EPS)
	{
//		cout << "\nk = " << k << ", Off(A) = " << ofd << endl;

		for (int t=0; t<n-1; t++)
		{
			for (int l=0; l<n/2; l++)
			{
				p = (top[l] < bot[l]) ? top[l] : bot[l];
				q = (top[l] > bot[l]) ? top[l] : bot[l];
//				cout << "p = " << p << ", q = " << q << endl;

				// Generate Givens rotation (c,s)
				Sym_schur2(n,a,p,q,&c,&s);
//				cout << "c = " << c << ", s = " << s << endl;

				// Apply Givens rotation
				Givens(n,a,p,q,c,s);

				// Apply Givens rotation from Right
				GivensR(n,v,p,q,c,s);

				k++;
			}
			music(n,top,bot);
		}
	}

	time = omp_get_wtime() - time;

	cout << "n = " << n << ", time = " << time << endl;
	cout << "k = " << k << ", Off(A) = " << Off_d(n,a) << endl;
	cout << "|| V^T A V - Λ|| = " << Residure(n,oa,a,v) << endl;

	delete [] a;
	delete [] v;
	delete [] oa;
	delete [] top;
	delete [] bot;

	return 0;
}
