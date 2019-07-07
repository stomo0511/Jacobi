//
// Parallel Order Jabobi
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
	int *top = new int[n/2];
	int *bot = new int[n/2];

	Gen_symmat(n,a); // Generate symmetric random matrix

	#pragma omp parallel
	{
		Copy_symmat(n,a,b);  // b <- a
		Set_Iden(n,v);       // v <- I

		#pragma omp for
		for (int i=0;i<n/2;i++)      // Index pair
		{
			top[i] = i*2;
			bot[i] = i*2+1;
		}
	}

	const double tol = n*sqrt(n)*e; // Convergence criteria
	int k = 0;                       // No. of iterations

	double time = omp_get_wtime();  // Timer start

	while (Off_d(n,a) > tol)
	{
		for (int j=0; j<n-1; j++)
		{
			for (int i=0; i<n/2; i++)
			{
				double c, s;  // Cos and Sin

				int p = (top[i] < bot[i]) ? top[i] : bot[i];
				int q = (top[i] > bot[i]) ? top[i] : bot[i];

				// Generate Givens rotation (c,s)
				Sym_schur2(n,a,p,q,&c,&s);

				// Apply Givens rotation from Left and Right
				GivensL(n,a,p,q,c,s);
				GivensR(n,a,p,q,c,s);

				// Apply Givens rotation from Right
				GivensR(n,v,p,q,c,s);

				k++;
			} // End of i-loop
			music(n,top,bot);
		} // End of j-loop
	} // End of while-loop

	time = omp_get_wtime() - time;  // Timer stop

	cout << "n = " << n << ", time = " << time << endl;
	cout << "k = " << k << ", Off(A) = " << Off_d(n,a) << endl;
	cout << "|| V^T A V - Λ|| = " << Residure(n,b,a,v) << endl;

	delete [] a;
	delete [] b;
	delete [] v;
	delete [] top;
	delete [] bot;

	return EXIT_SUCCESS;
}
