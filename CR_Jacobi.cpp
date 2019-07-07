//
// Cyclic-by-Row Jabobi
//

#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <limits>
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

	const double tol = n*sqrt(n)*e; // Convergence criteria
	double c, s;                    // Cos and Sin
	int k = 0;                       // No. of iterations

	double time = omp_get_wtime();  // Timer start

	while (Off_d(n,a) > tol)
	{
		for (int i=0; i<n; i++)
			for (int j=0; j<i; j++)
			{
				// Generate Givens rotation (c,s)
				Sym_schur2(n,a,i,j,&c,&s);

				// Apply Givens rotation from Left and Right
				GivensL(n,a,i,j,c,s);
				GivensR(n,a,i,j,c,s);

				// Apply Givens rotation from Right
				GivensR(n,v,i,j,c,s);

				k++;
			} // End of j-loop
		// End of i-loop
	} // End of while-loop

	time = omp_get_wtime() - time;  // Timer stop

	cout << "n = " << n << ", time = " << time << endl;
	cout << "k = " << k << ", Off(A) = " << Off_d(n,a) << endl;

	cout << "|| V^T A V - Λ|| = " << Residure(n,b,a,v) << endl;

	delete [] a;
	delete [] b;
	delete [] v;

	return EXIT_SUCCESS;
}
