#include <Python.h>

static PyObject*
fatode_erk_fwd_integrate(PyObject* self, PyObject* args)
{
  int nvar;
  double t_0, t_f, rel_tol, abs_tol;
  PyObject *initial_state;
  PyObject *future_state;
  PyObject *stage_vectors;
  PyObject *model;
  PyObject *work_vector;

  printf("Inside FATODE C Layer\n");

  // Parse all the arguments passed in from Python
  if(!PyArg_ParseTuple(args, "iOOddOddOO", &nvar, &initial_state, &future_state,
            &t_0, &t_f, &stage_vectors, &rel_tol, &abs_tol, &model, &work_vector))
  {
    PyErr_SetString(PyExc_TypeError, "Expected nvar, initial_state, future_state, t_0, t_f, K, rel_tol, abs_tol, model, work_vector");
    return NULL;
  }

  // Check that the list of stage_vectors is a list
  if(!PyList_Check(stage_vectors))
  {
    PyErr_SetString(PyExc_TypeError, "Expected a list object for stages");
    return NULL;
  }

  // if(!PyCallable_Check(rhsCallback))
  // {
  //   PyErr_SetString(PyExc_TypeError, "Expected a callback object for rhs");
  //   return NULL;
  // }

  // Call the fortran code.
  Py_XINCREF(initial_state);
  Py_XINCREF(future_state);
  Py_XINCREF(stage_vectors);

  Py_XINCREF(model);            // Expect this to be non null
  Py_XINCREF(work_vector);

  printf("Calling Fortran Layer\n");

  __erk_f90_integrator_MOD_integrate(&nvar, &t_0, &t_f, initial_state, future_state,
                                     stage_vectors, &rel_tol, &abs_tol, model,
                                     work_vector);
  printf("Leaving Fortran Layer\n");

  Py_XDECREF(initial_state);
  Py_XDECREF(future_state);
  Py_XDECREF(stage_vectors);
  Py_XDECREF(model);
  Py_XDECREF(work_vector);
  Py_INCREF(Py_None);
  return Py_None;
}

void c_ode_rhs_inlist_(PyObject *model, double *t, PyObject *in_state,
  PyObject *out_state_list, int *index)
{
  PyObject *stVec;

  Py_XINCREF(out_state_list);

  // Check that the list of state_vectors is a list
  if(!PyList_Check(out_state_list))
  {
    printf("Expected a list object for out state vectors");
    exit(1);
  }

  // Get the element
  stVec = PyList_GetItem(out_state_list, (*index) - 1);

  Py_XINCREF(stVec);

  // if(!PyCallable_Check(rhsCallback))
  // {
  //   printf("%s\n", );("Expected a callback object for rhs");
  //   exit(1);
  // }

  Py_XINCREF(model);            // Expect this to be non null
  Py_XINCREF(in_state);

  // Y <- Y + A * X
  PyObject_CallMethod(model, "step_forward_function", "(dOO)", *t, in_state, stVec);

  Py_XDECREF(stVec);
  Py_XDECREF(model);            // Expect this to be non null
  Py_XDECREF(in_state);
  Py_XDECREF(out_state_list);
}

void c_set2zero_(PyObject *stVec)
{
  Py_XINCREF(stVec);

  // Call the scale object and pass 0.0 method in the state vector object
  PyObject_CallMethod(stVec, "scale", "(d)", 0.0);

  Py_XDECREF(stVec);
}

void c_set2zero_inlist_(PyObject *list_of_stVec, int *index)
{
  PyObject *stVec;

  Py_XINCREF(list_of_stVec);

  // Check that the list of state_vectors is a list
  if(!PyList_Check(list_of_stVec))
  {
    printf("Expected a list object for state vectors");
    exit(1);
  }

  // Get the element
  stVec = PyList_GetItem(list_of_stVec, (*index) - 1);

  Py_XINCREF(stVec);

  // Call Set2Zero
  c_set2zero_(stVec);

  Py_XDECREF(stVec);
  Py_XDECREF(list_of_stVec);
}

void c_lss_norm_(PyObject *stVec, double *norm2)
{
  PyObject *tmpNrm;

  Py_XINCREF(stVec);
  // Call the norm method in the state vector object
  tmpNrm = PyObject_CallMethod(stVec, "norm2", NULL);

  if(!PyFloat_Check(tmpNrm))
  {
    printf("Norm is expected to be a double type.\n");
    exit(1);
  }

  *norm2 = PyFloat_AsDouble(tmpNrm);
  Py_XDECREF(stVec);
}

void c_lss_norm_inlist_(PyObject *list_of_stVec, int *index, double *norm2)
{
  PyObject *stVec;

  Py_XINCREF(list_of_stVec);

  // Check that the list of state_vectors is a list
  if(!PyList_Check(list_of_stVec))
  {
    printf("Expected a list object for state vectors");
    exit(1);
  }

  // Get the element
  stVec = PyList_GetItem(list_of_stVec, (*index) - 1);

  Py_XINCREF(stVec);

  // Call LSS_Norm
  c_lss_norm_(stVec, norm2);

  Py_XDECREF(stVec);
  Py_XDECREF(list_of_stVec);
}

void c_lss_daxpy_(double *alpha, PyObject *X, PyObject *Y)
{
  Py_XINCREF(X);
  Py_XINCREF(Y);

  // Y <- Y + A * X
  PyObject_CallMethod(Y, "axpy", "(dO)", *alpha, X);

  Py_XDECREF(X);
  Py_XDECREF(Y);
}

void c_lss_daxpy_inlist_(double *alpha, PyObject *X_list, int *index, PyObject *Y)
{
  PyObject *stVec;

  Py_XINCREF(X_list);
  Py_XINCREF(Y);

  // Check that the list of state_vectors is a list
  if(!PyList_Check(X_list))
  {
    printf("Expected a list object for state vectors");
    exit(1);
  }

  // Get the element
  stVec = PyList_GetItem(X_list, (*index) - 1);

  Py_XINCREF(stVec);

  // Y <- Y + A * X
  PyObject_CallMethod(Y, "axpy", "(dO)", *alpha, stVec);

  Py_XDECREF(stVec);
  Py_XDECREF(X_list);
  Py_XDECREF(Y);
}

static PyMethodDef int_method[] = {
    {"integrate",  fatode_erk_fwd_integrate,  METH_VARARGS,
        PyDoc_STR("Wrapper for FWD ERK from FatODE")},
    {NULL, NULL, 0, NULL}           /* sentinel */
};

PyMODINIT_FUNC
initfatode_erk_fwd(void)
{
  (void) Py_InitModule("fatode_erk_fwd", int_method);
}
