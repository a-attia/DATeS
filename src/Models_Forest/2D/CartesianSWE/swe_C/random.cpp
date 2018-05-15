
#include "random.h"
#include <random>
#include <chrono>
using namespace std;

default_random_engine *random_gen;
normal_distribution<double> *ndist;

void random_init(double mean, double stddev) {
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    random_gen = new default_random_engine(seed);
    ndist = new normal_distribution<double>(mean, stddev);
}

double random_gaussian() {
    return (*ndist)(*random_gen);
}

void random_cleanup() {
    delete random_gen;
    delete ndist;
}

