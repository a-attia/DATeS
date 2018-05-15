
#ifdef __cplusplus
extern "C" {
#endif

// Global variable
extern double X[];

// Functions
void swe_ic_base(double h, double u, double v);
void swe_ic_gaussian(int variable, double amplitude, int x0, int y0, double sigma_x, double sigma_y);
void swe_ic_rectangle(int variable, double amplitude, int x0, int y0, int len_x, int len_y);
void swe_ic_circle(int variable, double amplitude, int x0, int y0, int radius);
void swe_ic_sinusoid(int variable, double amplitude, double frequency);
void swe_ic_radial_velocity(double amplitude, double radius, int x0, int y0);
void swe_ic_rand_perturbations(double mean, double stddev);
void swe_ic_set(int select);

#ifdef __cplusplus
}
#endif

