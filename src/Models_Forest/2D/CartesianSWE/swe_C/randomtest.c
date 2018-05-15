
#include "random.h"
#include <stdio.h>
#define NB 10

int main() {
    unsigned int nrolls = 1000 * NB;
    unsigned int nstars = 100;
    unsigned int buckets[NB] = {};
    unsigned int i, j;

    random_init(double(NB)/2.0, double(NB)/5.0);

    for (i = 0; i < nrolls; i++) {
        double number = random_gaussian();
        if ((number >= 0.0) && (number < double(NB))) buckets[int(number)]++;
    }

    printf("normal distribution (%f, %f):\n", double(NB)/2.0, double(NB)/5.0);
    for (i = 0; i < NB; i++) {
        printf("%d - %d: ", i, i+1);
        for (j = 0; j < buckets[i]*nstars/nrolls; j++) {
            printf("*");
        }
        printf("\n");
    }

    random_cleanup();

    return 0;
}

