//
// Clasiccal Jacobi method
//

#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <limits>
#include <mkl_cblas.h>
#include "Jacobi.hpp"

using namespace std;

int main(int argc, char **argv)
{
	assert(argc > 1);  // Usage: a.out [size of matrix]

	constexpr double e = std::numeric_limits<double>::epsilon();  // Machine epsilon

	const int n = atoi(argv[1]);	    // Matrix size: n x n
	assert( n % 2 == 0);

	double *a = new double[n*n];   // Original matrix
	double *b = new double[n*n];   // Copy of original matrix
	double *v = new double[n*n];   // Right-singular vector matrix


	Gen_symmat(n,a); // Generate symmetric random matrix

	#pragma omp parallel
	{
		Copy_symmat(n,a,b);  // b <- a
		Set_Iden(n,v);       // v <- I
	}

	const double tol = sqrt(n)*e;  // Convergence criteria
	int k = 0;                       // No. of iterations
	double c, s;                    // Cos and Sin
	int p, q;                        // Index

	double time = omp_get_wtime();  // Timer start

	while (Off_d(n,a) > EPS)
	{
		Search_max(n,a,&p,&q);     // Search the maximum element

		Sym_schur2(n,a,p,q,&c,&s); // Generate Givens rotation (c,s)

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
//			cout << b[i + j*n] << ", ";
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
	cout << "|| V^T A V - Λ|| = " << Residure(n,b,a,v) << endl;

	delete [] a;
	delete [] v;
	delete [] b;

	return 0;
}
