//
// One sided Jacobi
//

#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>

#include <mkl_cblas.h>

#include "Jacobi.hpp"

using namespace std;

void printAtA(const int n, double* a)
{
	double tmp;

	for (int i=0; i<n; i++) {
		for (int j=0; j<n; j++) {
			tmp = 0.0;
			for (int k=0; k<n; k++)
				tmp += a[k + i*n]*a[k + j*n];
			cout << tmp << ", ";
		}
		cout << endl;
	}
	cout << endl;
}

void printA(const int n, double* a)
{
	for (int i=0; i<n; i++) {
		for (int j=0; j<n; j++)
			cout << a[i + j*n] << ", ";
		cout << endl;
	}
	cout << endl;
}

int main(int argc, char **argv)
{
	assert(argc > 1);

	// Matrix size
	const int n = atoi(argv[1]);
	assert(n%2==0);

	double *a = new double[n*n];

	// Generate matrix
	Gen_mat(n,a);

	// Singular vector matrix
	double *v = new double[n*n];

	#pragma omp parallel
	Set_Iden(n,v);

	int *top = new int[n/2];
	int *bot = new int[n/2];

	#pragma omp parallel for
	for (int i=0;i<n/2;i++)
	{
		top[i] = i*2;
		bot[i] = i*2+1;
	}

	// Print a
//	printAtA(n,a);

	int step = 0;

	double time = omp_get_wtime();

	double maxt = 1.0;  // convergence criterion
	while (maxt > EPS)
	{
		maxt = 0.0;
		for (int p=0; p<n-1; p++)
		{
			#pragma omp parallel for reduction (max:maxt)
			for (int l=0; l<n/2; l++)
			{
				int i = (top[l] > bot[l]) ? top[l] : bot[l];
				int j = (top[l] < bot[l]) ? top[l] : bot[l];
//				cout << "(i,j) = (" << i << ", " << j << ")\n";

				double x = cblas_ddot(n,a+i*n,1,a+i*n,1);  // x = a_i^T a_i
				double y = cblas_ddot(n,a+j*n,1,a+j*n,1);  // y = a_j^T a_j
				double z = cblas_ddot(n,a+i*n,1,a+j*n,1);  // z = a_i^T a_j

//				cout << "x = " << x << ", y = " << y << ", z = " << z << endl;
				double t = fabs(z) / sqrt(x*y);

				maxt = maxt > t ? maxt : t;

				if (t > EPS)
				{
					// compute rotation
					double zeta = (x - y) / (2.0*z);
					double tau;
					if (zeta >= 0.0)
						tau =  1.0 / ( zeta + sqrt(1.0 + zeta*zeta));
					else
						tau = -1.0 / (-zeta + sqrt(1.0 + zeta*zeta));

					double c = 1.0 / sqrt(1 + tau*tau);
					double s = c*tau;

					double tmp;

					// update A
					for (int k=0; k<n; k++)
					{
						tmp = a[k + i*n];
						a[k + i*n] =  c*tmp + s*a[k + j*n];
						a[k + j*n] = -s*tmp + c*a[k + j*n];
					}

					// update V
					for (int k=0; k<n; k++)
					{
						tmp = v[k + i*n];
						v[k + i*n] =  c*tmp + s*v[k + j*n];
						v[k + j*n] = -s*tmp + c*v[k + j*n];
					}
				}
				step++;
				// Print a
//				printAtA(n,a);
			}
			music(n,top,bot);
		}
//		cout << "maxt = " << maxt << endl;
	}

	time = omp_get_wtime() - time;

	cout << "n = " << n << ", time = " << time << endl;
	cout << "k = " << step << endl;

	delete [] a;
	delete [] v;

	return EXIT_SUCCESS;
}
