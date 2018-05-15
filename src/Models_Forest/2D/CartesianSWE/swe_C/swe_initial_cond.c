#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "swe_parameters.h"
#include "random.h"

// Global variable
double X[NVAR];

// Macros
#define Xdim (NX+2)
#define Ydim (NY+2)
#define H(I, J) ( (X)[(I) * Xdim + (J)] )
#define U(I, J) ( (X)[(Xdim * Ydim) + (I) * Xdim + (J)] )
#define V(I, J) ( (X)[(2 * Xdim * Ydim) + (I) * Xdim + (J)] )
#define PI (3.141592653589793)

// Functions
void swe_ic_base(double h, double u, double v) {
   int i, j;

   // Initialize height:
   for (i = 0; i < Ydim; i++) {
      for (j = 0; j < Xdim; j++) {
         H(i, j) = h;
      }
   }

   // Initialize x-direction velocities
   for (i = 0; i < Ydim; i++) {
      for (j = 0; j < Xdim; j++) {
         U(i, j) = u;
      }
   }

   // Initialize y-direction velocities
   for (i = 0; i < Ydim; i++) {
      for (j = 0; j < Xdim; j++) {
         V(i, j) = v;
      }
   }
}

void swe_ic_gaussian(int variable, double amplitude, int x0, int y0, double sigma_x, double sigma_y) {
   int i, j;
   double * matrix;
   double den_x, den_y, center_x, center_y;

   if (x0 < 0 || x0 >= NX*DX || y0 < 0 || y0 >= NY*DY) {
      printf("Initialization: Out of bounds center point: (%d, %d).\n", x0, y0);
      return;
   }

   switch(variable) {
      case 0:
         matrix = &H(0,0);
         break;
      case 1:
         matrix = &U(0,0);
         break;
      case 2:
         matrix = &V(0,0);
         break;
      default:
         printf("Initialization: Incorrect physical variable selected in gaussian, must be 0, 1 or 2: %d\n", variable);
         return;
   }

   den_x = 2.0 * (sigma_x / DX) * (sigma_x / DX);
   den_y = 2.0 * (sigma_y / DY) * (sigma_y / DY);
   center_x = 1.0 + x0 / DX;
   center_y = 1.0 + y0 / DY;

   for (i = 0; i < Ydim; i++) {
      for (j = 0; j < Xdim; j++) {
         matrix[i * Xdim + j] += amplitude * exp(-(pow(j - center_x, 2.0)/den_x + pow(i - center_y, 2.0)/den_y));
      }
   }
}

void swe_ic_rectangle(int variable, double amplitude, int x0, int y0, int len_x, int len_y) {
   int i, j;
   double * matrix;
   double center_x, center_y;
   int xlow, xhigh, ylow, yhigh;

   if (x0 < 0 || x0 >= NX*DX || y0 < 0 || y0 >= NY*DY) {
      printf("Initialization: Out of bounds center point: (%d, %d).\n", x0, y0);
      return;
   }

   switch(variable) {
      case 0:
         matrix = &H(0,0);
         break;
      case 1:
         matrix = &U(0,0);
         break;
      case 2:
         matrix = &V(0,0);
         break;
      default:
         printf("Initialization: Incorrect physical variable selected in rectangle, must be 0, 1 or 2: %d\n", variable);
         return;
   }

   // Center of rectangle
   center_x = 1.0 + x0 / DX;
   center_y = 1.0 + y0 / DX;

   // Boundaries of rectangle
   xlow = (int)(center_x - len_x/2.0);
   xhigh = (int)(center_x + len_x/2.0); 
   ylow = (int)(center_y - len_y/2.0);
   yhigh = (int)(center_y + len_y/2.0);

   // Limit rectangle to problem domain
   xlow = (xlow >= 0) ? xlow : 0;
   xhigh = (xhigh <= Xdim-1) ? xhigh : Xdim-1;
   ylow = (ylow >= 0) ? ylow : 0;
   yhigh = (yhigh <= Ydim-1) ? yhigh : Ydim-1;

   for (i = ylow; i < yhigh; i++) {
      for (j = xlow; j < xhigh; j++) {
         matrix[i * Xdim + j] += amplitude;
      }
   }
}

void swe_ic_circle(int variable, double amplitude, int x0, int y0, int radius) {
   int i, j;
   double * matrix;
   double center_x, center_y;

   if (x0 < 0 || x0 >= NX*DX || y0 < 0 || y0 >= NY*DY) {
      printf("Initialization: Out of bounds center point: (%d, %d).\n", x0, y0);
      return;
   }

   switch(variable) {
      case 0:
         matrix = &H(0,0);
         break;
      case 1:
         matrix = &U(0,0);
         break;
      case 2:
         matrix = &V(0,0);
         break;
      default:
         printf("Initialization: Incorrect physical variable selected in circle, must be 0, 1 or 2: %d\n", variable);
         return;
   }

   // Center of circle
   center_x = 1.0 + x0 / DX;
   center_y = 1.0 + y0 / DX;

   for (i = 0; i < Ydim; i++) {
      for (j = 0; j < Xdim; j++) {
         if (pow(j - center_x, 2.0) + pow(i - center_y, 2.0) <= radius * radius)
            matrix[i * Xdim + j] += amplitude;
      }
   }
}

void swe_ic_sinusoid(int variable, double amplitude, double frequency) {
   int i, j;
   double * matrix;
   double dist, rad_freq = PI*frequency;

   switch(variable) {
      case 0:
         matrix = &H(0,0);
         break;
      case 1:
         matrix = &U(0,0);
         break;
      case 2:
         matrix = &V(0,0);
         break;
      default:
         printf("Initialization: Incorrect physical variable selected in circle, must be 0, 1 or 2: %d\n", variable);
         return;
   }

   for (i = 0; i < Ydim; i++) {
      dist = (double)i / (double)Ydim;
      for (j = 0; j < Xdim; j++) {
         matrix[i * Xdim + j] += amplitude*(1.0 - dist)*sin(rad_freq * dist);
      }
   }
}

void swe_ic_radial_velocity(double amplitude, double radius, int x0, int y0) {
   int i, j;
   double r, r0, center_x, center_y;

   center_x = 1.0 + x0 / DX;
   center_y = 1.0 + y0 / DY;
   r0 = radius / fmax(DX, DY);

   for (i = 0; i < Ydim; i++) {
      for (j = 0; j < Xdim; j++) {
         r = fmax(sqrt(pow(i - center_y, 2.0) + pow(j - center_x, 2.0)), 1.0);
         U(i, j) += fmax(amplitude * (1.0 - r / r0) * r, 0.0) * (double)(i - center_x) / Xdim;
         V(i, j) += fmax(amplitude * (1.0 - r / r0) * r, 0.0) * (double)(j - center_y) / Ydim;
      }
   }
}

void swe_ic_rand_perturbations(double mean, double stddev) {
   unsigned int i;

   random_init(mean, stddev);

   for (i = 0; i < NVAR; i++) {
      X[i] += 0.01 * X[i] * random_gaussian();
   }
}

void swe_ic_set(int select) {
   switch(select) {
      case 0:
         swe_ic_base(1.0, 0.0, 0.0);
         swe_ic_gaussian(0, 0.7, 13, 11, 5.0, 5.0);
         swe_ic_gaussian(0, 0.3, 23, 25, 2.1, 2.1);
         swe_ic_gaussian(0, -0.2, 17, 15, 3.1, 2.9); // 2.894);
         break;
      case 1:
         swe_ic_base(1.0, 0.0, 0.0);
         swe_ic_rectangle(1, -0.5, 10, 10, 8, 8);
         swe_ic_rectangle(1, 0.5, 20, 10, 8, 8);
         swe_ic_rectangle(1, 1.0, 20, 15, 8, 4);
         swe_ic_rectangle(1, -0.5, 10, 20, 8, 8);
         swe_ic_rectangle(1, 0.5, 20, 20, 8, 8);
         swe_ic_rectangle(1, -1.0, 10, 15, 8, 4);

         swe_ic_rectangle(2, -0.5, 10, 10, 8, 8);
         swe_ic_rectangle(2, -1.0, 15, 10, 8, 4);
         swe_ic_rectangle(2, -0.5, 20, 10, 8, 8);
         swe_ic_rectangle(2, 0.5, 10, 20, 8, 8);
         swe_ic_rectangle(2, 1.0, 15, 20, 8, 4);
         swe_ic_rectangle(2, 0.5, 20, 20, 8, 8);
         break;
      case 2:
         // Whirlpool:
         swe_ic_base(1.0, 0.0, 0.0);
         swe_ic_rectangle(1, 1.0, 20, 10, 8, 8);
         swe_ic_rectangle(1, -1.0, 10, 20, 8, 8);
         swe_ic_rectangle(2, -1.0, 10, 10, 8, 8);
         swe_ic_rectangle(2, 1.0, 20, 20, 8, 8);

         swe_ic_gaussian(0, -0.5, 15, 15, 3.0, 3.0);
         break;
      case 3:
         swe_ic_base(1.0, 0.0, 0.0);
         swe_ic_rectangle(0, 0.8, 3, 16, 12, 20);
         swe_ic_gaussian(0, 1.5, 11,  6, 6.0, 2.0);
         swe_ic_gaussian(0, 1.5, 11, 16, 6.0, 2.0);
         swe_ic_gaussian(0, 1.5, 11, 27, 6.0, 2.0);
         swe_ic_gaussian(0, -1.0, 14,  7, 7.0, 2.85);
         swe_ic_gaussian(0, -1.0, 14, 17, 3.0, 2.85);
         swe_ic_gaussian(0, -1.0, 14, 26, 7.0, 2.85);
         swe_ic_rectangle(0, -0.8, 27, 16, 12, 20);

         swe_ic_rectangle(1, 0.8, 7, 16, 12, 20);
         break;
      case 4:
         swe_ic_base(1.0e3, 0.0, 0.0);
         swe_ic_sinusoid(0, 0.7e3, 2.0);
         break;
      case 5:
         swe_ic_base(1.0e4, 0.0, 0.0);
         swe_ic_gaussian(0, 0.7, 13, 11, 5.0, 5.0);
         swe_ic_gaussian(0, 0.3, 23, 25, 2.1, 2.1);
         swe_ic_gaussian(0, -0.2, 17, 15, 3.1, 2.9); // 2.894);
         break;
      case 6:
         swe_ic_base(1.0e3, 0.0, 0.0);
         swe_ic_gaussian(0, 0.8e3, 16, 16, 2.0, 2.0);
         swe_ic_radial_velocity(1.0e1, 12.0, 16, 16);
         break;
      case 7:
         swe_ic_base(1.0e4, 0.0, 0.0);
         swe_ic_radial_velocity(1.0e2, 10.0, 16, 16);
         break;
      case 8:
         swe_ic_base(1.0e4, 0.0, 0.0);
         swe_ic_radial_velocity(1.0, 10.0, 16, 16);
         break;
      case 9:
         swe_ic_base(1.0e8, 0.0, 0.0);
         swe_ic_radial_velocity(1.0e-2, 10.0, 16, 16);
         break;
      case 10:
         swe_ic_base(1.0, 0.0, 0.0);
         swe_ic_gaussian(0, 0.7, 13, 11, 5.0, 5.0);
         swe_ic_gaussian(0, 0.3, 23, 25, 2.1, 2.1);
         swe_ic_gaussian(0, -0.2, 17, 15, 3.1, 2.9); // 2.894);
         swe_ic_rand_perturbations(0.0, 1.0);
         break;
      default:
         printf("Unrecognized initial condition selection: %d\n", select);
         exit(-1);
   }
}


