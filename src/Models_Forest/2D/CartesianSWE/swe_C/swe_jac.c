
#include <stdio.h>
#include "swe_parameters.h"

#define LEN 40
#define NUMNONZEROS 50

void swe_jac(double t, double Y[], double J[]);
void swe_jac_vec(double J[], double vector[], double product[]);
// void swe_jac_full(double J[], double J_full[]);
void swe_jac_main(int M, int N, double X[], double dx, double dy, int ROW[], int COL[], double J[]);

int J_ROW[NSPARSE];
int J_COL[NSPARSE];

void swe_jac(double t, double Y[], double J[]) {
   
   swe_jac_main(NX+2, NY+2, Y, DX, DY, J_ROW, J_COL, J);
}

void swe_jac_vec(double J[], double vector[], double product[]) {
   unsigned long i;
   
   for (i = 0; i < NVAR; i++) product[i] = 0.0;
   
   for (i = 0; i < NSPARSE; i++) {
      product[J_ROW[i]] += J[i] * vector[J_COL[i]];
   }
}

// void swe_jac_full(double J[], double J_full[]) {
//    unsigned long i;
//    
//    for (i = 0; i < NVAR*NVAR; i++)
//       J_full[i] = 0.0;
//    
//    for (i = 0; i < NSPARSE; i++) {
//       J_full[J_ROW[i]*NVAR + J_COL[i]] = J[i];
//    }
// }

void swe_jac_print(double J[]) {
   int i;
//    int j;
//    double J_full[NVAR][NVAR] = { {0.0, 0.0}, {0.0, 0.0} };
   FILE *j_output = fopen("swe_jac_output.dat", "w");
   
//    for (i = 0; i < NSPARSE; i++) {
//       J_full[J_ROW[i]][J_COL[i]] = J[i];
//    }
//    
//    for (i = 0; i < NVAR; i++) {
//       for (j = 0; j < NVAR; j++) {
//          fprintf(j_output, "%25.18e ", J_full[i][j]);
//       }
//    }
   
   for (i = 0; i < NSPARSE; i++) {
      fprintf(j_output, " %25.18e  %d %d\n", J[i], J_ROW[i], J_COL[i]);
   }
   
   fclose(j_output);
}

void swe_jac_main(int M, int N, double X[], double dx, double dy, int ROW[], int COL[], double J[])
{
		unsigned long i,j,l,sparseLength, tmp,m,boundary_len, tmp1,n;
		double a1,a2,a3,a4,a5;
		double a6,a7,a8,a9,a10, a11,a12;
		
		l=M*N;
		sparseLength=l*LEN;
		boundary_len=(2*(M+N)-4)*NUMNONZEROS;
		unsigned long col[sparseLength], row[sparseLength],row1[boundary_len],col1[boundary_len];
		double val[sparseLength],val1[boundary_len];
		tmp=-1;
		tmp1=-1;
		m=M;
		n=N;
		/*printf ("%d %d\n",M,N);*/
		for(i=0;i<M-2;i++)
		{
				for(j=0;j<N-2;j++)
				{
						unsigned long k = (i+1)*M+j+1;
						tmp=tmp+1;
						row[tmp]=k;
						col[tmp]=k+l+M;
						val[tmp]=-1/(2*dx);
						tmp=tmp+1;
						row[tmp]=k;
						col[tmp]=k+l-m;
						val[tmp]=1/(2*dx);
						tmp = tmp+1;
						row[tmp]=k;
						col[tmp]=k+2*l+1;
						val[tmp]=-1/(2*dy);
						tmp=tmp+1;
						row[tmp]=k;
						col[tmp]=k+2*l-1;
						val[tmp]=1/(2*dy);
						tmp=tmp+1;
						if(i==0 || j==0)
						{
								if(i ==0 && j==0)
								{
										tmp1=tmp1+1;
										row1[tmp1]=k-n-1;
										col1[tmp1]=k+l+m;
										val1[tmp1]=-1/(2*dx);
										tmp1=tmp1+1;
										row1[tmp1]=k-n-1;
										col1[tmp1]=k+l-m;
										val1[tmp1]=1/(2*dx);
										tmp1=tmp1+1;
										row1[tmp1]=k-n-1;
										col1[tmp1]=k+2*l+1;
										val1[tmp1]=-1/(2*dy);
										tmp1=tmp1+1;
										row1[tmp1]=k-n-1;
										col1[tmp1]=k+2*l-1;
										val1[tmp1]=1/(2*dy);
								}
						
								if(i==0)
								{
										tmp1=tmp1+1;
										row1[tmp1]=k-n;
										col1[tmp1]=k+l+m;
										val1[tmp1]=-1/(2*dx);
										tmp1=tmp1+1;
										row1[tmp1]=k-n;
										col1[tmp1]=k+l-m;
										val1[tmp1]=1/(2*dx);
										tmp1=tmp1+1;
										row1[tmp1]=k-n;
										col1[tmp1]=k+2*l+1;
										val1[tmp1]=-1/(2*dy);
										tmp1=tmp1+1;
										row1[tmp1]=k-n;
										col1[tmp1]=k+2*l-1;
										val1[tmp1]=1/(2*dy);
								}
								if(j==0)
								{
										tmp1=tmp1+1;
										row1[tmp1]=k-1;
										col1[tmp1]=k+l+m;
										val1[tmp1]=-1/(2*dx);
										tmp1=tmp1+1;
										row1[tmp1]=k-1;
										col1[tmp1]=k+l-m;
										val1[tmp1]=1/(2*dx);
										tmp1=tmp1+1;
										row1[tmp1]=k-1;
										col1[tmp1]=k+2*l+1;
										val1[tmp1]=-1/(2*dy);
										tmp1=tmp1+1;
										row1[tmp1]=k-1;
										col1[tmp1]=k+2*l-1;
										val1[tmp1]=1/(2*dy);
								}
								if(i==0 && j==N-3)
								{
										tmp1=tmp1+1;
										row1[tmp1]=k-n+1;
										col1[tmp1]=k+l+m;
										val1[tmp1]=-1/(2*dx);
										tmp1=tmp1+1;
										row1[tmp1]=k-n+1;
										col1[tmp1]=k+l-m;
										val1[tmp1]=1/(2*dx);
										tmp1=tmp1+1;
										row1[tmp1]=k-n+1;
										col1[tmp1]=k+2*l+1;
										val1[tmp1]=-1/(2*dy);
										tmp1=tmp1+1;
										row1[tmp1]=k-n+1;
										col1[tmp1]=k+2*l-1;
										val1[tmp1]=1/(2*dy);
								}
								if(i==M-3 && j==0)
								{
										tmp1=tmp1+1;
										row1[tmp1]=k+n-1;
										col1[tmp1]=k+l+m;
										val1[tmp1]=-1/(2*dx);
										tmp1=tmp1+1;
										row1[tmp1]=k+n-1;
										col1[tmp1]=k+l-m;
										val1[tmp1]=1/(2*dx);
										tmp1=tmp1+1;
										row1[tmp1]=k+n-1;
										col1[tmp1]=k+2*l+1;
										val1[tmp1]=-1/(2*dy);
										tmp1=tmp1+1;
										row1[tmp1]=k+n-1;
										col1[tmp1]=k+2*l-1;
										val1[tmp1]=1/(2*dy);

								}
						}
						if(i==M-3 ||j==N-3)
						{
								if(i==M-3 && j== N-3)
								{
										tmp1=tmp1+1;
										row1[tmp1]=k+n+1;
										col1[tmp1]=k+l+m;
										val1[tmp1]=-1/(2*dx);
										tmp1=tmp1+1;
										row1[tmp1]=k+n+1;
										col1[tmp1]=k+l-m;
										val1[tmp1]=1/(2*dx);
										tmp1=tmp1+1;
										row1[tmp1]=k+n+1;
										col1[tmp1]=k+2*l+1;
										val1[tmp1]=-1/(2*dy);
										tmp1=tmp1+1;
										row1[tmp1]=k+n+1;
										col1[tmp1]=k+2*l-1;
										val1[tmp1]=1/(2*dy);
								}
								if(i==M-3)
								{
										tmp1=tmp1+1;
										row1[tmp1]=k+n;
										col1[tmp1]=k+l+m;
										val1[tmp1]=-1/(2*dx);
										tmp1=tmp1+1;
										row1[tmp1]=k+n;
										col1[tmp1]=k+l-m;
										val1[tmp1]=1/(2*dx);
										tmp1=tmp1+1;
										row1[tmp1]=k+n;
										col1[tmp1]=k+2*l+1;
										val1[tmp1]=-1/(2*dy);
										tmp1=tmp1+1;
										row1[tmp1]=k+n;
										col1[tmp1]=k+2*l-1;
										val1[tmp1]=1/(2*dy);
								}
								if(j==N-3)
								{
										tmp1=tmp1+1;
										row1[tmp1]=k+1;
										col1[tmp1]=k+l+m;
										val1[tmp1]=-1/(2*dx);
										tmp1=tmp1+1;
										row1[tmp1]=k+1;
										col1[tmp1]=k+l-m;
										val1[tmp1]=1/(2*dx);
										tmp1=tmp1+1;
										row1[tmp1]=k+1;
										col1[tmp1]=k+2*l+1;
										val1[tmp1]=-1/(2*dy);
										tmp1=tmp1+1;
										row1[tmp1]=k+1;
										col1[tmp1]=k+2*l-1;
										val1[tmp1]=1/(2*dy);
								}
						}
						k=k+l;
						a1 = (X[k+m]+X[k])/(X[k-l+m]+X[k-l]); /*X(50)+X(44)..*/
						a2 = (X[k]+X[k-m])/(X[k-l]+X[k-l-m]); /*(44)+X(38)..*/
						a3 = (X[k+l+1]+X[k+l])/(X[k-l+1]+X[k-l]); /*%x(81)+x(80)..*/
						a4 = (X[k+l-1]+X[k+l])/(X[k-l-1]+X[k-l]); /*%x(79)+x(80)..*/
						a5 = (X[k-l+m]+X[k-l]); /*%x(14)+x(8)*/
						a6 = (X[k-l]+X[k-l-m]); /*%x(8)+x(2)*/
						a7 = (X[k+1]+X[k])/(X[k-l+1]+X[k-l]); /*%x(45)+x(44)..*/
						a8 = (X[k]+X[k-1])/(X[k-l]+X[k-l-1]); /*%x(44)+x(43)..*/
						row[tmp]=k;
						col[tmp]=k+m;
						val[tmp]=-a1/dx;
						tmp=tmp+1;
						row[tmp]=k;
						col[tmp]=k;
						val[tmp]= (a1-a2)*(-1/dx)-(1/(2*dy))*(a3-a4);
						tmp=tmp+1;
						row[tmp]=k;
						col[tmp]=k-l+m;
						val[tmp] = (-1/dx)*(-a1*a1/2+g/4*a5);
						tmp=tmp+1;
						row[tmp]=k;
						col[tmp]=k-l;
						val[tmp] = (-1/dx)*((-a1*a1/2+g/4*a5)-(-a2*a2/2+g/4*a6))- (1/dy)*(-a3*a7/2 +a4*a8/2);
						tmp=tmp+1;
						row[tmp]=k;
						col[tmp]=k-m;
						val[tmp] = a2/(dx);
						tmp=tmp+1;
						row[tmp]=k;
						col[tmp]=k-l-m;
						val[tmp] = (-1/dx)*(a2*a2/2-g/4*a6);
						tmp=tmp+1;
						row[tmp]=k;
						col[tmp]=k+l+1;
						val[tmp] = -a7/(2*dy);
						tmp=tmp+1;
						row[tmp]=k;
						col[tmp]=k+l;
						val[tmp] = (-1/dy)*(a7/2-a8/2);
						tmp=tmp+1;
						row[tmp]=k;
						col[tmp]=k+1;
						val[tmp] = (-1/dy)*(a3/2);
						tmp=tmp+1;
						row[tmp]=k;
						col[tmp]=k+l-1;
						val[tmp]=(1/dy)*a8/2;
						tmp=tmp+1;
						row[tmp]=k;
						col[tmp]=k-1;
						val[tmp] = a4/(2*dy);
						tmp=tmp+1;
						row[tmp]=k;
						col[tmp]=k-l-1;
						val[tmp] = (-1/dy)*(a4*a8/2);
						tmp=tmp+1;
						row[tmp]=k;
						col[tmp]=k-l+1;
						val[tmp] = (1/dy)*(a3*a7/2);
						tmp=tmp+1;
						if(i==0||j==0)
						{
								if(i==0 && j==0)
								{
										tmp1=tmp1+1;
										row1[tmp1]=k-n-1;
										col1[tmp1]=k+m;
										val1[tmp1]=-a1/dx*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n-1;
										col1[tmp1]=k;
										val1[tmp1]= ((a1-a2)*(-1/dx)-(1/(2*dy))*(a3-a4))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n-1;
										col1[tmp1]=k-l+m;
										val1[tmp1] = ((-1/dx)*(-a1*a1/2+g/4*a5))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n-1;
										col1[tmp1]=k-l;
										val1[tmp1] = ((-1/dx)*((-a1*a1/2+g/4*a5)-(-a2*a2/2+g/4*a6))- (1/dy)*(-a3*a7/2 +a4*a8/2))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n-1;
										col1[tmp1]=k-m;
										val1[tmp1] = (a2/(dx))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n-1;
										col1[tmp1]=k-l-m;
										val1[tmp1] = ((-1/dx)*(a2*a2/2-g/4*a6))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n-1;
										col1[tmp1]=k+l+1;
										val1[tmp1] = (-a7/(2*dy))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n-1;
										col1[tmp1]=k+l;
										val1[tmp1] = ((-1/dy)*(a7/2-a8/2))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n-1;
										col1[tmp1]=k+1;
										val1[tmp1] = ((-1/dy)*(a3/2))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n-1;
										col1[tmp1]=k+l-1;
										val1[tmp1]=(1/dy)*a8/2*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n-1;
										col1[tmp1]=k-1;
										val1[tmp1] = a4/(2*dy)*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n-1;
										col1[tmp1]=k-l-1;
										val1[tmp1] = ((-1/dy)*(a4*a8/2))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n-1;
										col1[tmp1]=k-l+1;
										val1[tmp1] = ((1/dy)*(a3*a7/2))*-1;
								}
						

								if(i==0)
								{
										tmp1=tmp1+1;
										row1[tmp1]=k-n;
										col1[tmp1]=k+m;
										val1[tmp1]=-a1/dx*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n;
										col1[tmp1]=k;
										val1[tmp1]= ((a1-a2)*(-1/dx)-(1/(2*dy))*(a3-a4))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n;
										col1[tmp1]=k-l+m;
										val1[tmp1] = ((-1/dx)*(-a1*a1/2+g/4*a5))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n;
										col1[tmp1]=k-l;
										val1[tmp1] = ((-1/dx)*((-a1*a1/2+g/4*a5)-(-a2*a2/2+g/4*a6))- (1/dy)*(-a3*a7/2 +a4*a8/2))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n;
										col1[tmp1]=k-m;
										val1[tmp1] = (a2/(dx))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n;
										col1[tmp1]=k-l-m;
										val1[tmp1] = ((-1/dx)*(a2*a2/2-g/4*a6))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n;
										col1[tmp1]=k+l+1;
										val1[tmp1] = (-a7/(2*dy))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n;
										col1[tmp1]=k+l;
										val1[tmp1] = ((-1/dy)*(a7/2-a8/2))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n;
										col1[tmp1]=k+1;
										val1[tmp1] = ((-1/dy)*(a3/2))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n;
										col1[tmp1]=k+l-1;
										val1[tmp1]=(1/dy)*a8/2*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n;
										col1[tmp1]=k-1;
										val1[tmp1] = a4/(2*dy)*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n;
										col1[tmp1]=k-l-1;
										val1[tmp1] = ((-1/dy)*(a4*a8/2))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n;
										col1[tmp1]=k-l+1;
										val1[tmp1] = ((1/dy)*(a3*a7/2))*-1;

								}
								if(j==0)
								{
										tmp1=tmp1+1;
										row1[tmp1]=k-1;
										col1[tmp1]=k+m;
										val1[tmp1]=-a1/dx;
										tmp1=tmp1+1;
										row1[tmp1]=k-1;
										col1[tmp1]=k;
										val1[tmp1]= (a1-a2)*(-1/dx)-(1/(2*dy))*(a3-a4);
										tmp1=tmp1+1;
										row1[tmp1]=k-1;
										col1[tmp1]=k-l+m;
										val1[tmp1] = (-1/dx)*(-a1*a1/2+g/4*a5);
										tmp1=tmp1+1;
										row1[tmp1]=k-1;
										col1[tmp1]=k-l;
										val1[tmp1] = (-1/dx)*((-a1*a1/2+g/4*a5)-(-a2*a2/2+g/4*a6))- (1/dy)*(-a3*a7/2 +a4*a8/2);
										tmp1=tmp1+1;
										row1[tmp1]=k-1;
										col1[tmp1]=k-m;
										val1[tmp1] = a2/(dx);
										tmp1=tmp1+1;
										row1[tmp1]=k-1;
										col1[tmp1]=k-l-m;
										val1[tmp1] = (-1/dx)*(a2*a2/2-g/4*a6);
										tmp1=tmp1+1;
										row1[tmp1]=k-1;
										col1[tmp1]=k+l+1;
										val1[tmp1] = -a7/(2*dy);
										tmp1=tmp1+1;
										row1[tmp1]=k-1;
										col1[tmp1]=k+l;
										val1[tmp1] = (-1/dy)*(a7/2-a8/2);
										tmp1=tmp1+1;
										row1[tmp1]=k-1;
										col1[tmp1]=k+1;
										val1[tmp1] = (-1/dy)*(a3/2);
										tmp1=tmp1+1;
										row1[tmp1]=k-1;
										col1[tmp1]=k+l-1;
										val1[tmp1]=(1/dy)*a8/2;
										tmp1=tmp1+1;
										row1[tmp1]=k-1;
										col1[tmp1]=k-1;
										val1[tmp1] = a4/(2*dy);
										tmp1=tmp1+1;
										row1[tmp1]=k-1;
										col1[tmp1]=k-l-1;
										val1[tmp1] = (-1/dy)*(a4*a8/2);
										tmp1=tmp1+1;
										row1[tmp1]=k-1;
										col1[tmp1]=k-l+1;
										val1[tmp1] = (1/dy)*(a3*a7/2);

								}
								if(i==0 && j==N-3)
								{
										tmp1=tmp1+1;
										row1[tmp1]=k-n+1;
										col1[tmp1]=k+m;
										val1[tmp1]=-a1/dx*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n+1;
										col1[tmp1]=k;
										val1[tmp1]= ((a1-a2)*(-1/dx)-(1/(2*dy))*(a3-a4))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n+1;
										col1[tmp1]=k-l+m;
										val1[tmp1] = ((-1/dx)*(-a1*a1/2+g/4*a5))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n+1;
										col1[tmp1]=k-l;
										val1[tmp1] = ((-1/dx)*((-a1*a1/2+g/4*a5)-(-a2*a2/2+g/4*a6))- (1/dy)*(-a3*a7/2 +a4*a8/2))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n+1;
										col1[tmp1]=k-m;
										val1[tmp1] = (a2/(dx))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n+1;
										col1[tmp1]=k-l-m;
										val1[tmp1] = ((-1/dx)*(a2*a2/2-g/4*a6))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n+1;
										col1[tmp1]=k+l+1;
										val1[tmp1] = (-a7/(2*dy))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n+1;
										col1[tmp1]=k+l;
										val1[tmp1] = ((-1/dy)*(a7/2-a8/2))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n+1;
										col1[tmp1]=k+1;
										val1[tmp1] = ((-1/dy)*(a3/2))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n+1;
										col1[tmp1]=k+l-1;
										val1[tmp1]=(1/dy)*a8/2*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n+1;
										col1[tmp1]=k-1;
										val1[tmp1] = a4/(2*dy)*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n+1;
										col1[tmp1]=k-l-1;
										val1[tmp1] = ((-1/dy)*(a4*a8/2))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n+1;
										col1[tmp1]=k-l+1;
										val1[tmp1] = ((1/dy)*(a3*a7/2))*-1;
								}
								if(i==M-3 && j==0)
								{
										tmp1=tmp1+1;
										row1[tmp1]=k+n-1;
										col1[tmp1]=k+m;
										val1[tmp1]=-a1/dx*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n-1;
										col1[tmp1]=k;
										val1[tmp1]= ((a1-a2)*(-1/dx)-(1/(2*dy))*(a3-a4))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n-1;
										col1[tmp1]=k-l+m;
										val1[tmp1] = ((-1/dx)*(-a1*a1/2+g/4*a5))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n-1;
										col1[tmp1]=k-l;
										val1[tmp1] = ((-1/dx)*((-a1*a1/2+g/4*a5)-(-a2*a2/2+g/4*a6))- (1/dy)*(-a3*a7/2 +a4*a8/2))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n-1;
										col1[tmp1]=k-m;
										val1[tmp1] = (a2/(dx))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n-1;
										col1[tmp1]=k-l-m;
										val1[tmp1] = ((-1/dx)*(a2*a2/2-g/4*a6))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n-1;
										col1[tmp1]=k+l+1;
										val1[tmp1] = (-a7/(2*dy))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n-1;
										col1[tmp1]=k+l;
										val1[tmp1] = ((-1/dy)*(a7/2-a8/2))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n-1;
										col1[tmp1]=k+1;
										val1[tmp1] = ((-1/dy)*(a3/2))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n-1;
										col1[tmp1]=k+l-1;
										val1[tmp1]=(1/dy)*a8/2*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n-1;
										col1[tmp1]=k-1;
										val1[tmp1] = a4/(2*dy)*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n-1;
										col1[tmp1]=k-l-1;
										val1[tmp1] = ((-1/dy)*(a4*a8/2))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n-1;
										col1[tmp1]=k-l+1;
										val1[tmp1] = ((1/dy)*(a3*a7/2))*-1;
								}


						}
						if(i==M-3 || j==N-3)
						{
								if(i==M-3&&j==N-3)
								{
										tmp1=tmp1+1;
										row1[tmp1]=k+n+1;
										col1[tmp1]=k+m;
										val1[tmp1]=-a1/dx*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n+1;
										col1[tmp1]=k;
										val1[tmp1]= ((a1-a2)*(-1/dx)-(1/(2*dy))*(a3-a4))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n+1;
										col1[tmp1]=k-l+m;
										val1[tmp1] = ((-1/dx)*(-a1*a1/2+g/4*a5))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n+1;
										col1[tmp1]=k-l;
										val1[tmp1] = ((-1/dx)*((-a1*a1/2+g/4*a5)-(-a2*a2/2+g/4*a6))- (1/dy)*(-a3*a7/2 +a4*a8/2))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n+1;
										col1[tmp1]=k-m;
										val1[tmp1] = (a2/(dx))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n+1;
										col1[tmp1]=k-l-m;
										val1[tmp1] = ((-1/dx)*(a2*a2/2-g/4*a6))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n+1;
										col1[tmp1]=k+l+1;
										val1[tmp1] = (-a7/(2*dy))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n+1;
										col1[tmp1]=k+l;
										val1[tmp1] = ((-1/dy)*(a7/2-a8/2))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n+1;
										col1[tmp1]=k+1;
										val1[tmp1] = ((-1/dy)*(a3/2))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n+1;
										col1[tmp1]=k+l-1;
										val1[tmp1]=(1/dy)*a8/2*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n+1;
										col1[tmp1]=k-1;
										val1[tmp1] = a4/(2*dy)*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n+1;
										col1[tmp1]=k-l-1;
										val1[tmp1] = ((-1/dy)*(a4*a8/2))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n+1;
										col1[tmp1]=k-l+1;
										val1[tmp1] = ((1/dy)*(a3*a7/2))*-1;
								}
								if(i==M-3)
								{
										tmp1=tmp1+1;
										row1[tmp1]=k+n;
										col1[tmp1]=k+m;
										val1[tmp1]=-a1/dx*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n;
										col1[tmp1]=k;
										val1[tmp1]= ((a1-a2)*(-1/dx)-(1/(2*dy))*(a3-a4))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n;
										col1[tmp1]=k-l+m;
										val1[tmp1] = ((-1/dx)*(-a1*a1/2+g/4*a5))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n;
										col1[tmp1]=k-l;
										val1[tmp1] = ((-1/dx)*((-a1*a1/2+g/4*a5)-(-a2*a2/2+g/4*a6))- (1/dy)*(-a3*a7/2 +a4*a8/2))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n;
										col1[tmp1]=k-m;
										val1[tmp1] = (a2/(dx))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n;
										col1[tmp1]=k-l-m;
										val1[tmp1] = ((-1/dx)*(a2*a2/2-g/4*a6))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n;
										col1[tmp1]=k+l+1;
										val1[tmp1] = (-a7/(2*dy))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n;
										col1[tmp1]=k+l;
										val1[tmp1] = ((-1/dy)*(a7/2-a8/2))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n;
										col1[tmp1]=k+1;
										val1[tmp1] = ((-1/dy)*(a3/2))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n;
										col1[tmp1]=k+l-1;
										val1[tmp1]=(1/dy)*a8/2*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n;
										col1[tmp1]=k-1;
										val1[tmp1] = a4/(2*dy)*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n;
										col1[tmp1]=k-l-1;
										val1[tmp1] = ((-1/dy)*(a4*a8/2))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n;
										col1[tmp1]=k-l+1;
										val1[tmp1] = ((1/dy)*(a3*a7/2))*-1;
								}
								if(j==N-3)
								{
										tmp1=tmp1+1;
										row1[tmp1]=k+1;
										col1[tmp1]=k+m;
										val1[tmp1]=-a1/dx;
										tmp1=tmp1+1;
										row1[tmp1]=k+1;
										col1[tmp1]=k;
										val1[tmp1]= (a1-a2)*(-1/dx)-(1/(2*dy))*(a3-a4);
										tmp1=tmp1+1;
										row1[tmp1]=k+1;
										col1[tmp1]=k-l+m;
										val1[tmp1] = (-1/dx)*(-a1*a1/2+g/4*a5);
										tmp1=tmp1+1;
										row1[tmp1]=k+1;
										col1[tmp1]=k-l;
										val1[tmp1] = (-1/dx)*((-a1*a1/2+g/4*a5)-(-a2*a2/2+g/4*a6))- (1/dy)*(-a3*a7/2 +a4*a8/2);
										tmp1=tmp1+1;
										row1[tmp1]=k+1;
										col1[tmp1]=k-m;
										val1[tmp1] = a2/(dx);
										tmp1=tmp1+1;
										row1[tmp1]=k+1;
										col1[tmp1]=k-l-m;
										val1[tmp1] = (-1/dx)*(a2*a2/2-g/4*a6);
										tmp1=tmp1+1;
										row1[tmp1]=k+1;
										col1[tmp1]=k+l+1;
										val1[tmp1] = -a7/(2*dy);
										tmp1=tmp1+1;
										row1[tmp1]=k+1;
										col1[tmp1]=k+l;
										val1[tmp1] = (-1/dy)*(a7/2-a8/2);
										tmp1=tmp1+1;
										row1[tmp1]=k+1;
										col1[tmp1]=k+1;
										val1[tmp1] = (-1/dy)*(a3/2);
										tmp1=tmp1+1;
										row1[tmp1]=k+1;
										col1[tmp1]=k+l-1;
										val1[tmp1]=(1/dy)*a8/2;
										tmp1=tmp1+1;
										row1[tmp1]=k+1;
										col1[tmp1]=k-1;
										val1[tmp1] = a4/(2*dy);
										tmp1=tmp1+1;
										row1[tmp1]=k+1;
										col1[tmp1]=k-l-1;
										val1[tmp1] = (-1/dy)*(a4*a8/2);
										tmp1=tmp1+1;
										row1[tmp1]=k+1;
										col1[tmp1]=k-l+1;
										val1[tmp1] = (1/dy)*(a3*a7/2);
								}


						}
						k = k+l;
						a9 = (X[k+m]+X[k])/(X[k-2*l+m]+X[k-2*l]); /*%x(80)+x(86)..*/
						a10 =(X[k]+X[k-m])/(X[k-2*l]+X[k-2*l-m]); /*%x(80)+x(74)..*/
						a11 =(X[k-2*l]+X[k-2*l+1]); /*%x(9)+x(8)*/
						a12 =(X[k-2*l]+X[k-2*l-1]); /*%x(8)+x(7)*/
						row[tmp]=k;
						col[tmp]=k-l+m;
						val[tmp] = -a9/(2*dx);
						tmp=tmp+1;
						row[tmp]=k;
						col[tmp]=k-l;
						val[tmp] = (-1/(2*dx))*(a9-a10);
						tmp=tmp+1;
						row[tmp]=k;
						col[tmp]=k;
						val[tmp] = (-1/(2*dx))*(a1-a2)-(1/dy)*(a3-a4);
						tmp=tmp+1;
						row[tmp]=k;
						col[tmp]=k+m;
						val[tmp] = -a1/(2*dx);
						tmp=tmp+1;
						row[tmp]=k;
						col[tmp]=k-2*l+m;
						val[tmp] = (a1*a9)/(2*dx);
						tmp=tmp+1;
						row[tmp]=k;
						col[tmp]=k-2*l;
						val[tmp] = (-1/dx)*(-a1*a9/2+a2*a10/2)-(1/dy)*(-a3*a3/2+g/4*(a11)+a4*a4/2-g/4*a12);
						tmp=tmp+1;
						row[tmp]=k;
						col[tmp]=k-l-m;
						val[tmp] = (a10/(2*dx));
						tmp=tmp+1;
						row[tmp]=k;
						col[tmp]=k-m;
						val[tmp] = (a2/(2*dx));
						tmp=tmp+1;
						row[tmp]=k;
						col[tmp]=k-2*l-m;
						val[tmp] = (-1/(2*dx))*(a2*a10);
						tmp=tmp+1;
						row[tmp]=k;
						col[tmp]=k+1;
						val[tmp] = -a3/dy;
						tmp=tmp+1;
						row[tmp]=k;
						col[tmp]=k-1;
						val[tmp] = a4/dy;
						tmp=tmp+1;
						row[tmp]=k;
						col[tmp]=k-2*l+1;
						val[tmp] = (-1/dy)*(-a3*a3/2+g/4*a11);
						tmp=tmp+1;
						row[tmp]=k;
						col[tmp]=k-2*l-1;
						val[tmp] = (-1/dy)*(a4*a4/2-g/4*a12);
						if(i==0 || j==0)
						{
								if(i==0 && j==0)
								{
										tmp1=tmp1+1;
										row1[tmp1]=k-n-1;
										col1[tmp1]=k-l+m;
										val1[tmp1] = -a9/(2*dx)*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n-1;
										col1[tmp1]=k-l;
										val1[tmp1] = ((-1/(2*dx))*(a9-a10))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n-1;
										col1[tmp1]=k;
										val1[tmp1] = ((-1/(2*dx))*(a1-a2)-(1/dy)*(a3-a4))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n-1;
										col1[tmp1]=k+m;
										val1[tmp1] = (-a1/(2*dx))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n-1;
										col1[tmp1]=k-2*l+m;
										val1[tmp1] = ((a1*a9)/(2*dx))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n-1;
										col1[tmp1]=k-2*l;
										val1[tmp1] = ((-1/dx)*(-a1*a9/2+a2*a10/2)-(1/dy)*(-a3*a3/2+g/4*(a11)+a4*a4/2-g/4*a12))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n-1;
										col1[tmp1]=k-l-m;
										val1[tmp1] = ((a10/(2*dx)))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n-1;
										col1[tmp1]=k-m;
										val1[tmp1] = ((a2/(2*dx)))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n-1;
										col1[tmp1]=k-2*l-m;
										val1[tmp1] = ((-1/(2*dx))*(a2*a10))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n-1;
										col1[tmp1]=k+1;
										val1[tmp1] = (-a3/dy)*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n-1;
										col1[tmp1]=k-1;
										val1[tmp1] = (a4/dy)*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n-1;
										col1[tmp1]=k-2*l+1;
										val1[tmp1] = ((-1/dy)*(-a3*a3/2+g/4*a11))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n-1;
										col1[tmp1]=k-2*l-1;
										val1[tmp1] = ((-1/dy)*(a4*a4/2-g/4*a12))*-1;

								}
								if(i==0)
								{
										tmp1=tmp1+1;
										row1[tmp1]=k-n;
										col1[tmp1]=k-l+m;
										val1[tmp1] = -a9/(2*dx);
										tmp1=tmp1+1;
										row1[tmp1]=k-n;
										col1[tmp1]=k-l;
										val1[tmp1] = (-1/(2*dx))*(a9-a10);
										tmp1=tmp1+1;
										row1[tmp1]=k-n;
										col1[tmp1]=k;
										val1[tmp1] = (-1/(2*dx))*(a1-a2)-(1/dy)*(a3-a4);
										tmp1=tmp1+1;
										row1[tmp1]=k-n;
										col1[tmp1]=k+m;
										val1[tmp1] = -a1/(2*dx);
										tmp1=tmp1+1;
										row1[tmp1]=k-n;
										col1[tmp1]=k-2*l+m;
										val1[tmp1] = (a1*a9)/(2*dx);
										tmp1=tmp1+1;
										row1[tmp1]=k-n;
										col1[tmp1]=k-2*l;
										val1[tmp1] = (-1/dx)*(-a1*a9/2+a2*a10/2)-(1/dy)*(-a3*a3/2+g/4*(a11)+a4*a4/2-g/4*a12);
										tmp1=tmp1+1;
										row1[tmp1]=k-n;
										col1[tmp1]=k-l-m;
										val1[tmp1] = (a10/(2*dx));
										tmp1=tmp1+1;
										row1[tmp1]=k-n;
										col1[tmp1]=k-m;
										val1[tmp1] = (a2/(2*dx));
										tmp1=tmp1+1;
										row1[tmp1]=k-n;
										col1[tmp1]=k-2*l-m;
										val1[tmp1] = (-1/(2*dx))*(a2*a10);
										tmp1=tmp1+1;
										row1[tmp1]=k-n;
										col1[tmp1]=k+1;
										val1[tmp1] = -a3/dy;
										tmp1=tmp1+1;
										row1[tmp1]=k-n;
										col1[tmp1]=k-1;
										val1[tmp1] = a4/dy;
										tmp1=tmp1+1;
										row1[tmp1]=k-n;
										col1[tmp1]=k-2*l+1;
										val1[tmp1] = (-1/dy)*(-a3*a3/2+g/4*a11);
										tmp1=tmp1+1;
										row1[tmp1]=k-n;
										col1[tmp1]=k-2*l-1;
										val1[tmp1] = (-1/dy)*(a4*a4/2-g/4*a12);
								}
								if(j==0)
								{
										tmp1=tmp1+1;
										row1[tmp1]=k-1;
										col1[tmp1]=k-l+m;
										val1[tmp1] = (-a9/(2*dx))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-1;
										col1[tmp1]=k-l;
										val1[tmp1] = ((-1/(2*dx))*(a9-a10))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-1;
										col1[tmp1]=k;
										val1[tmp1] = ((-1/(2*dx))*(a1-a2)-(1/dy)*(a3-a4))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-1;
										col1[tmp1]=k+m;
										val1[tmp1] = (-a1/(2*dx))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-1;
										col1[tmp1]=k-2*l+m;
										val1[tmp1] = ((a1*a9)/(2*dx))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-1;
										col1[tmp1]=k-2*l;
										val1[tmp1] = ((-1/dx)*(-a1*a9/2+a2*a10/2)-(1/dy)*(-a3*a3/2+g/4*(a11)+a4*a4/2-g/4*a12))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-1;
										col1[tmp1]=k-l-m;
										val1[tmp1] = ((a10/(2*dx)))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-1;
										col1[tmp1]=k-m;
										val1[tmp1] = ((a2/(2*dx)))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-1;
										col1[tmp1]=k-2*l-m;
										val1[tmp1] = ((-1/(2*dx))*(a2*a10))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-1;
										col1[tmp1]=k+1;
										val1[tmp1] = (-a3/dy)*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-1;
										col1[tmp1]=k-1;
										val1[tmp1] = (a4/dy)*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-1;
										col1[tmp1]=k-2*l+1;
										val1[tmp1] = ((-1/dy)*(-a3*a3/2+g/4*a11))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-1;
										col1[tmp1]=k-2*l-1;
										val1[tmp1] = ((-1/dy)*(a4*a4/2-g/4*a12))*-1;
											
								}
								if(i==0 && j==N-3)
								{
										tmp1=tmp1+1;
										row1[tmp1]=k-n+1;
										col1[tmp1]=k-l+m;
										val1[tmp1] = -a9/(2*dx)*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n+1;
										col1[tmp1]=k-l;
										val1[tmp1] = ((-1/(2*dx))*(a9-a10))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n+1;
										col1[tmp1]=k;
										val1[tmp1] = ((-1/(2*dx))*(a1-a2)-(1/dy)*(a3-a4))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n+1;
										col1[tmp1]=k+m;
										val1[tmp1] = (-a1/(2*dx))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n+1;
										col1[tmp1]=k-2*l+m;
										val1[tmp1] = ((a1*a9)/(2*dx))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n+1;
										col1[tmp1]=k-2*l;
										val1[tmp1] = ((-1/dx)*(-a1*a9/2+a2*a10/2)-(1/dy)*(-a3*a3/2+g/4*(a11)+a4*a4/2-g/4*a12))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n+1;
										col1[tmp1]=k-l-m;
										val1[tmp1] = ((a10/(2*dx)))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n+1;
										col1[tmp1]=k-m;
										val1[tmp1] = ((a2/(2*dx)))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n+1;
										col1[tmp1]=k-2*l-m;
										val1[tmp1] = ((-1/(2*dx))*(a2*a10))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n+1;
										col1[tmp1]=k+1;
										val1[tmp1] = (-a3/dy)*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n+1;
										col1[tmp1]=k-1;
										val1[tmp1] = (a4/dy)*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n+1;
										col1[tmp1]=k-2*l+1;
										val1[tmp1] = ((-1/dy)*(-a3*a3/2+g/4*a11))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k-n+1;
										col1[tmp1]=k-2*l-1;
										val1[tmp1] = ((-1/dy)*(a4*a4/2-g/4*a12))*-1;
								}
								if(i==N-3 && j==0)
								{

										tmp1=tmp1+1;
										row1[tmp1]=k+n-1;
										col1[tmp1]=k-l+m;
										val1[tmp1] = -a9/(2*dx)*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n-1;
										col1[tmp1]=k-l;
										val1[tmp1] = ((-1/(2*dx))*(a9-a10))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n-1;
										col1[tmp1]=k;
										val1[tmp1] = ((-1/(2*dx))*(a1-a2)-(1/dy)*(a3-a4))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n-1;
										col1[tmp1]=k+m;
										val1[tmp1] = (-a1/(2*dx))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n-1;
										col1[tmp1]=k-2*l+m;
										val1[tmp1] = ((a1*a9)/(2*dx))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n-1;
										col1[tmp1]=k-2*l;
										val1[tmp1] = ((-1/dx)*(-a1*a9/2+a2*a10/2)-(1/dy)*(-a3*a3/2+g/4*(a11)+a4*a4/2-g/4*a12))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n-1;
										col1[tmp1]=k-l-m;
										val1[tmp1] = ((a10/(2*dx)))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n-1;
										col1[tmp1]=k-m;
										val1[tmp1] = ((a2/(2*dx)))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n-1;
										col1[tmp1]=k-2*l-m;
										val1[tmp1] = ((-1/(2*dx))*(a2*a10))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n-1;
										col1[tmp1]=k+1;
										val1[tmp1] = (-a3/dy)*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n-1;
										col1[tmp1]=k-1;
										val1[tmp1] = (a4/dy)*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n-1;
										col1[tmp1]=k-2*l+1;
										val1[tmp1] = ((-1/dy)*(-a3*a3/2+g/4*a11))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n-1;
										col1[tmp1]=k-2*l-1;
										val1[tmp1] = ((-1/dy)*(a4*a4/2-g/4*a12))*-1;

								}

						}
						if(i==M-3 || j==N-3)
						{
								if(i==M-3 && j==N-3)
								{
										tmp1=tmp1+1;
										row1[tmp1]=k+n+1;
										col1[tmp1]=k-l+m;
										val1[tmp1] = (-a9/(2*dx))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n+1;
										col1[tmp1]=k-l;
										val1[tmp1] = ((-1/(2*dx))*(a9-a10))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n+1;
										col1[tmp1]=k;
										val1[tmp1] = ((-1/(2*dx))*(a1-a2)-(1/dy)*(a3-a4))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n+1;
										col1[tmp1]=k+m;
										val1[tmp1] = (-a1/(2*dx))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n+1;
										col1[tmp1]=k-2*l+m;
										val1[tmp1] = ((a1*a9)/(2*dx))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n+1;
										col1[tmp1]=k-2*l;
										val1[tmp1] = ((-1/dx)*(-a1*a9/2+a2*a10/2)-(1/dy)*(-a3*a3/2+g/4*(a11)+a4*a4/2-g/4*a12))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n+1;
										col1[tmp1]=k-l-m;
										val1[tmp1] = ((a10/(2*dx)))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n+1;
										col1[tmp1]=k-m;
										val1[tmp1] = ((a2/(2*dx)))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n+1;
										col1[tmp1]=k-2*l-m;
										val1[tmp1] = ((-1/(2*dx))*(a2*a10))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n+1;
										col1[tmp1]=k+1;
										val1[tmp1] = (-a3/dy)*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n+1;
										col1[tmp1]=k-1;
										val1[tmp1] = (a4/dy)*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n+1;
										col1[tmp1]=k-2*l+1;
										val1[tmp1] = ((-1/dy)*(-a3*a3/2+g/4*a11))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+n+1;
										col1[tmp1]=k-2*l-1;
										val1[tmp1] = ((-1/dy)*(a4*a4/2-g/4*a12))*-1;
								}
								if(i==M-3)
								{
										tmp1=tmp1+1;
										row1[tmp1]=k+n;
										col1[tmp1]=k-l+m;
										val1[tmp1] = -a9/(2*dx);
										tmp1=tmp1+1;
										row1[tmp1]=k+n;
										col1[tmp1]=k-l;
										val1[tmp1] = (-1/(2*dx))*(a9-a10);
										tmp1=tmp1+1;
										row1[tmp1]=k+n;
										col1[tmp1]=k;
										val1[tmp1] = (-1/(2*dx))*(a1-a2)-(1/dy)*(a3-a4);
										tmp1=tmp1+1;
										row1[tmp1]=k+n;
										col1[tmp1]=k+m;
										val1[tmp1] = -a1/(2*dx);
										tmp1=tmp1+1;
										row1[tmp1]=k+n;
										col1[tmp1]=k-2*l+m;
										val1[tmp1] = (a1*a9)/(2*dx);
										tmp1=tmp1+1;
										row1[tmp1]=k+n;
										col1[tmp1]=k-2*l;
										val1[tmp1] = (-1/dx)*(-a1*a9/2+a2*a10/2)-(1/dy)*(-a3*a3/2+g/4*(a11)+a4*a4/2-g/4*a12);
										tmp1=tmp1+1;
										row1[tmp1]=k+n;
										col1[tmp1]=k-l-m;
										val1[tmp1] = (a10/(2*dx));
										tmp1=tmp1+1;
										row1[tmp1]=k+n;
										col1[tmp1]=k-m;
										val1[tmp1] = (a2/(2*dx));
										tmp1=tmp1+1;
										row1[tmp1]=k+n;
										col1[tmp1]=k-2*l-m;
										val1[tmp1] = (-1/(2*dx))*(a2*a10);
										tmp1=tmp1+1;
										row1[tmp1]=k+n;
										col1[tmp1]=k+1;
										val1[tmp1] = -a3/dy;
										tmp1=tmp1+1;
										row1[tmp1]=k+n;
										col1[tmp1]=k-1;
										val1[tmp1] = a4/dy;
										tmp1=tmp1+1;
										row1[tmp1]=k+n;
										col1[tmp1]=k-2*l+1;
										val1[tmp1] = (-1/dy)*(-a3*a3/2+g/4*a11);
										tmp1=tmp1+1;
										row1[tmp1]=k+n;
										col1[tmp1]=k-2*l-1;
										val1[tmp1] = (-1/dy)*(a4*a4/2-g/4*a12);

								}
								if(j==N-3)
								{
										tmp1=tmp1+1;
										row1[tmp1]=k+1;
										col1[tmp1]=k-l+m;
										val1[tmp1] = (-a9/(2*dx))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+1;
										col1[tmp1]=k-l;
										val1[tmp1] = ((-1/(2*dx))*(a9-a10))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+1;
										col1[tmp1]=k;
										val1[tmp1] = ((-1/(2*dx))*(a1-a2)-(1/dy)*(a3-a4))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+1;
										col1[tmp1]=k+m;
										val1[tmp1] = (-a1/(2*dx))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+1;
										col1[tmp1]=k-2*l+m;
										val1[tmp1] = ((a1*a9)/(2*dx))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+1;
										col1[tmp1]=k-2*l;
										val1[tmp1] = ((-1/dx)*(-a1*a9/2+a2*a10/2)-(1/dy)*(-a3*a3/2+g/4*(a11)+a4*a4/2-g/4*a12))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+1;
										col1[tmp1]=k-l-m;
										val1[tmp1] = ((a10/(2*dx)))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+1;
										col1[tmp1]=k-m;
										val1[tmp1] = ((a2/(2*dx)))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+1;
										col1[tmp1]=k-2*l-m;
										val1[tmp1] = ((-1/(2*dx))*(a2*a10))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+1;
										col1[tmp1]=k+1;
										val1[tmp1] = (-a3/dy)*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+1;
										col1[tmp1]=k-1;
										val1[tmp1] = (a4/dy)*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+1;
										col1[tmp1]=k-2*l+1;
										val1[tmp1] = ((-1/dy)*(-a3*a3/2+g/4*a11))*-1;
										tmp1=tmp1+1;
										row1[tmp1]=k+1;
										col1[tmp1]=k-2*l-1;
										val1[tmp1] = ((-1/dy)*(a4*a4/2-g/4*a12))*-1;
								}
						}

				}
		}
		/*printf("%d\n", tmp);*/
		tmp=tmp+1;
		tmp1=tmp1+1;
		/*printf("tmp1=%d\n",tmp1);*/
		
		/*printf("Before assignment\n");*/
		for(i=0;i<tmp;i++)
		{
				ROW[i]=row[i];
				COL[i]=col[i];
				J[i]=val[i];
		}
		for(i=0;i<tmp1;i++)
		{
				ROW[tmp+i]=row1[i];
				COL[tmp+i]=col1[i];
				J[tmp+i]=val1[i];
		}

		return;
}









