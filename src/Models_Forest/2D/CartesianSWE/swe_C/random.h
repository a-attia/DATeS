
#ifdef __cplusplus
extern "C" {
#endif

void random_init(double mean, double stddev);
double random_gaussian(void);
void   random_cleanup(void);

#ifdef __cplusplus
}
#endif

