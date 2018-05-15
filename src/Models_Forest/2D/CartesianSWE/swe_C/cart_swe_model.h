

// Functions to be called directly by the Python class.

// Model class functions
int model_init(int mesh_size, int thread_count, double *initial_cond);
int model_del();
int model_rhs(double t, double *in_vector, double *out_vector);

// State matrix functions
int model_jac_vec(double t, double *in_state, double *in_vector, double *out_vector);
int model_jac_t_vec(double t, double *in_state, double *in_vector, double *out_vector);

// State vector functions
double * vec_init();
int vec_del(double *in_vector);
int vec_get_size(int *out_integer);
int vec_scale(double in_alpha, double *io_vector);
int vec_copy(double *in_vector, double *out_vector);
int vec_dot(double *in_vector_x, double *in_vector_y, double *out_scalar);
int vec_axpy(double in_alpha, double *in_vector_x, double *io_vector_y);
int vec_add(double *in_vector_x, double *io_vector_y);
int vec_norm2(double *in_vector, double *out_scalar);

