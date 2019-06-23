#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <cmath>

using namespace std;

// Generate matrix: n=3
//void Gen_mat(const int n, double *a)
//{
//	a[0 + 0*n] = 1.0;
//	a[0 + 1*n] = a[1 + 0*n] = 2.0;
//	a[0 + 2*n] = a[2 + 0*n] = 3.0;
//	a[1 + 1*n] = 4.0;
//	a[2 + 1*n] = a[1 + 2*n] = 5.0;
//	a[2 + 2*n] = 6.0;
//}

// Generate matrix: n=4
//void Gen_mat(const int n, double *a)
//{
//	a[0 + 0*n] = a[0 + 1*n] = a[0 + 2*n] = a[0 + 3*n] = 1.0;
//	a[1 + 0*n] = 1.0; a[1 + 1*n] = 2.0; a[1 + 2*n] = 3.0; a[1 + 3*n] = 4.0;
//	a[2 + 0*n] = 1.0; a[2 + 1*n] = 3.0; a[2 + 2*n] = 6.0; a[2 + 3*n] = 10.0;
//	a[3 + 0*n] = 1.0; a[3 + 1*n] = 4.0; a[3 + 2*n] = 10.0; a[3 + 3*n] = 20.0;
//}

void Gen_mat(const int n, double *a)
{
	srand(20190611);

//	#pragma omp parallel for
	for (int i=0; i<n; i++)
		for (int j=0; j<=i; j++)
			a[i + j*n] = a[j + i*n] = (double)rand() / RAND_MAX;
}

void Copy_mat(const int n, double *a, double *b)
{
//	#pragma omp parallel for
	#pragma omp for
	for (int i=0; i<n; i++)
		for (int j=0; j<=i; j++)
			b[i + j*n] = b[j + i*n] = a[i + j*n];
}

void Set_Iden(const int n, double *a)
{
	#pragma omp parallel for
	for (int i=0; i<n; i++)
		for (int j=0; j<=i; j++)
			(i != j) ? a[i + j*n] = a[j + i*n] = 0.0: a[i + j*n] = 1.0;
}

// Norm of the off-diagonal elements
double Off_d(const int n, double *a)
{
	double tmp = 0.0;

	#pragma omp parallel for reduction(+:tmp)
	for (int i=0; i<n; i++)
		for (int j=0; j<n; j++)
			if (j != i)
				tmp += a[i + j*n]*a[i + j*n];
	return sqrt(tmp);
}

// Generate Givens rotation (c,s) to eliminate the (p,q) and (q,p) elements
void Sym_schur2(const int n, double *a, const int p, const int q, double *c, double *s)
{
	if (a[p + q*n] != 0.0)
	{
		double tau = (a[q + q*n] - a[p + p*n]) / (2.0*a[p + q*n]);
		double t;
		if (tau >= 0.0) {
			t =  1.0 / ( tau + sqrt(1.0+tau*tau));
		} else {
			t = -1.0 / (-tau + sqrt(1.0+tau*tau));
		}
		*c = 1.0 / sqrt(1+t*t);
		*s = t*(*c);
	}
	else
	{
		*c = 1.0; *s = 0.0;
	}
}

// Search element which has the maximum abs. val.
void Search_max(const int n, double *a, int *p, int *q)
{
	double tmp = 0.0;
	for (int i=1; i<n; i++)
		for (int j=0; j<i; j++)
			if (tmp < fabs(a[i + j*n])) {
				tmp = fabs(a[i + j*n]);
				*p = i;
				*q = j;
			}
}

// Search element which has the maximum abs. val.
// Parallel version
void PSearch_max(const int n, double *a, int *p, int *q)
{
	int t = omp_get_max_threads();
	double *tmp = new double [t];
	int *pv = new int [t];
	int *qv = new int [t];

	#pragma parallel
	{
		#pragma omp for
		for (int i=0; i<t; i++)
			tmp[i] = 0.0;

		#pragma omp for
		for (int i=1; i<n; i++)
		{
			int tid = omp_get_thread_num();
			for (int j=0; j<i; j++)
				if (tmp[tid] < fabs(a[i + j*n])) {
					tmp[tid] = fabs(a[i + j*n]);
					pv[tid] = i;
					qv[tid] = j;
			}
		}
	}

	double max = 0.0;
	for (int i=0; i<t; i++)
		if (max < tmp[i]) {
			*p = pv[i]; *q = qv[i];
		}

	delete [] tmp;
	delete [] pv;
	delete [] qv;
}

// Apply Givens rotation from Left and Right
void Givens(const int n, double *a, const int p, const int q, const double c, const double s)
{
	double app = a[p + p*n];
	double aqq = a[q + q*n];
	double apq = a[p + q*n];
	double tmp;

	for(int k=0; k<n; k++)
	{
		tmp = c*a[p + k*n] -s*a[q + k*n];
		a[q + k*n] = a[k + q*n] = s*a[p + k*n] +c*a[q + k*n];
		a[p + k*n] = a[k + p*n] = tmp;
	}
	a[p + q*n] = a[q + p*n] = 0.0;
	a[p + p*n] = c*c*app + s*s*aqq -2.0*c*s*apq;
	a[q + q*n] = s*s*app + c*c*aqq +2.0*c*s*apq;
}

// Apply Givens rotation from Right
void GivensR(const int n, double *v, const int p, const int q, const double c, const double s)
{
	double tmp;

	for(int k=0; k<n; k++)
	{
		tmp = c*v[k + p*n] -s*v[k + q*n];
		v[k + q*n] = s*v[k + p*n] +c*v[k + q*n];
		v[k + p*n] = tmp;
	}
}

// Apply Givens rotation from Left
void GivensL(const int n, double *v, const int p, const int q, const double c, const double s)
{
	double tmp;

	for(int k=0; k<n; k++)
	{
		tmp = c*v[p + k*n] -s*v[q + k*n];
		v[q + k*n] = s*v[p + k*n] +c*v[q + k*n];
		v[p + k*n] = tmp;
	}
}

// Apply Givens rotation from Left and Right
void Givens2(const int n, double *a, const int p, const int q, const double c, const double s, double *b)
{
	double app = a[p + p*n];
	double aqq = a[q + q*n];
	double apq = a[p + q*n];

	for(int k=0; k<n; k++)
	{
		b[p + k*n] = b[k + p*n] = c*a[p + k*n] -s*a[q + k*n];
		b[q + k*n] = b[k + q*n] = s*a[p + k*n] +c*a[q + k*n];
	}
	b[p + q*n] = b[q + p*n] = 0.0;
	b[p + p*n] = c*c*app + s*s*aqq -2.0*c*s*apq;
	b[q + q*n] = s*s*app + c*c*aqq +2.0*c*s*apq;
}

// Apply Givens rotation from Right
void GivensR2(const int n, double *v, const int p, const int q, const double c, const double s, double *u)
{
	for(int k=0; k<n; k++)
	{
		u[k + p*n] = c*v[k + p*n] -s*v[k + q*n];
		u[k + q*n] = s*v[k + p*n] +c*v[k + q*n];
	}
}

// Apply Givens rotation from Left
void GivensL2(const int n, double *v, const int p, const int q, const double c, const double s, double *u)
{
	for(int k=0; k<n; k++)
	{
		u[p + k*n] = c*v[p + k*n] -s*v[q + k*n];
		u[q + k*n] = s*v[p + k*n] +c*v[q + k*n];
	}
}

void music(const int n, int *top, int *bot)
{
	int m = n/2;
	int *ct = new int[m];
	int *cb = new int[m];

	for (int i=0; i<m; i++)
	{
		ct[i] = top[i]; cb[i] = bot[i];
	}

	for (int k=0; k<m; k++)
	{
		if (k==0)
			top[0] = 0;
		else if (k==1)
			top[1] = cb[0];
		else if (k>1)
			top[k] = ct[k-1];
		if (k==m-1)
			bot[k] = ct[k];
		else
			bot[k] = cb[k+1];
	}
}

//
// || V^T A V - L ||
//
double Residure(const int n, double* oa, double* a, double *v)
{
	double tmp;
	double *l = new double [n];

	// Store eigenvalues to the vector l
	for (int i=0; i<n; i++)
		l[i] = a[i + i*n];

	// V^T A
	for (int i=0; i<n; i++)
		for (int j=0; j<n; j++) {
			tmp = 0.0;
			for (int k=0; k<n; k++)
				tmp += v[k + i*n] * oa[k + j*n];
			a[i + j*n] = tmp;
		}

	// V^T A V
	for (int i=0; i<n; i++)
		for (int j=0; j<n; j++) {
			tmp = 0.0;
			for (int k=0; k<n; k++)
				tmp += a[i + k*n] * v[k + j*n];
			oa[i + j*n] = tmp;
		}

	tmp = 0.0;
	for (int i=0; i<n; i++)
		tmp += (oa[i + i*n] - l[i]) * (oa[i + i*n] - l[i]);

	delete [] l;

	return sqrt(tmp);
}

