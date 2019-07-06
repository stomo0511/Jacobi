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

// update two vectors
void Update(const int n, double* a, const int p, const int q, const double c, const double s)
{
	double tmp;

	for (int k=0; k<n; k++)
	{
		tmp = a[k + p*n];
		a[k + p*n] =  c*tmp - s*a[k + q*n];
		a[k + q*n] =  s*tmp + c*a[k + q*n];
	}
}

int main(int argc, char **argv)
{
	assert(argc > 1);

	// Matrix size: n x n
	const int n = atoi(argv[1]);
	assert( n % 2 == 0);

	double *a = new double[n*n];   // Original matrix
	double *oa = new double[n*n];  // Copy of original matrix
	double *v = new double[n*n];   // Right-singular vector matrix
	int *top = new int[n/2];
	int *bot = new int[n/2];

	// Generate random number (-1,1) matrix
	Gen_mat(n,a);

	#pragma omp parallel
	{
		Copy_mat(n,a,oa);   // oa <- a
		Set_Iden(n,v);      // v <- I

		#pragma omp for
		for (int i=0;i<n/2;i++)
		{
			top[i] = i*2;
			bot[i] = i*2+1;
		}
	}

	int k = 0;   // no. of iterations
	double time = omp_get_wtime();

	double maxt = 1.0;  // maximum abs. value of non-diagonal elements
	while (maxt > EPS)
	{
		maxt = 0.0;
		for (int j=0; j<n-1; j++)
		{
			#pragma omp parallel for reduction(+:k)
			for (int i=0; i<n/2; i++)
			{
				int p = (top[i] > bot[i]) ? top[i] : bot[i];
				int q = (top[i] < bot[i]) ? top[i] : bot[i];

				double x = cblas_ddot(n,a+p*n,1,a+p*n,1);  // x = a_p^T a_p
				double y = cblas_ddot(n,a+q*n,1,a+q*n,1);  // y = a_q^T a_q
				double z = cblas_ddot(n,a+p*n,1,a+q*n,1);  // z = a_p^T a_q

				double t = fabs(z) / sqrt(x*y);
				maxt = maxt > t ? maxt : t;      // *** <- この部分大丈夫化か？ ***

				if (t > EPS)
				{
					// compute Givens rotation
					double zeta = (y - x) / (2.0*z);
					double tau;
					if (zeta >= 0.0)
						tau =  1.0 / ( zeta + sqrt(1.0 + zeta*zeta));
					else
						tau = -1.0 / (-zeta + sqrt(1.0 + zeta*zeta));

					double c = 1.0 / sqrt(1 + tau*tau);
					double s = c*tau;

					// update A
					Update(n,a,p,q,c,s);

					// update V
					Update(n,v,p,q,c,s);
				}
				k++;
			} // End of l-loop
			music(n,top,bot);
		} // End of p-loop
	} // End of while-loop

	time = omp_get_wtime() - time;

//	// sigma_i = || G(:,i) ||_2
//	double *s = new double[n];
//
//	#pragma omp parallel for
//	for (int i=0; i<n; i++)
//		s[i] = sqrt(cblas_ddot(n,a+i*n,1,a+i*n,1));
//
//	// Left-singular vector matrix
//	double *u = new double[n*n];
//
//	#pragma omp parallel
//	{
//		Copy_mat(n,a,u);
//
//		#pragma omp for
//		for (int i=0; i<n; i++)
//			cblas_dscal(n,1.0/s[i],u+i*n,1);
//	}

	cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,n,n,n,-1.0,a,n,v,n,1.0,oa,n);
	double tmp = 0.0;
	#pragma omp parallel for reduction(+:tmp)
	for (int i=0; i<n*n; i++)
		tmp += oa[i]*oa[i];

	cout << "n = " << n << ", time = " << time << endl;
	cout << "k = " << k << ", ||A - U Sigma V^T|| = " << sqrt(tmp) << endl;

	delete [] a;
	delete [] oa;
	delete [] v;
	delete [] top;
	delete [] bot;
//	delete [] s;
//	delete [] u;

	return EXIT_SUCCESS;
}
