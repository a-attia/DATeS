.. DATeS documentation master file, created by
   sphinx-quickstart on Sat Apr 15 00:44:08 2017.
   

=====================================================================================================


.. image:: DATeS_Logo.png
   :scale: 50 %
   :target: index.html
   :align: center
   

The **Data Assimilation Testing Suite** (DATeS) is aimed to be a unified testing suite for `data assimilation (DA)`_ applications where researchers can collaborate, so that it would be much easier to understand and compare different methodologies in different settings. The core of DATeS is implemented in Python to enable for Object-Oriented capabilities. The main functionalities, such as the models, the data assimilation algorithms, the linear algebra solvers, and the time discretization routines are independent of each other, such as to offer maximum flexibility to configure data assimilation applications. DATeS can interface easily with large third party models written in Fortran or C, and with various external solvers.

.. _data assimilation (DA): https://en.wikipedia.org/wiki/Data_assimilation

=====================================================================================================


First Steps with DATeS
=======================
.. toctree::
   :maxdepth: 1
   
   Introduction <Intro>
   
   Download <Download>
   
   Installation <Installation>
   
   Initialization <Initialization>
   

DATeS Tutorials
================
.. toctree::
   :maxdepth: 1
   
    Tutorials <Tutorials>

   
Package Contents
=================

.. toctree::
   :maxdepth: 1

    Linear Algebra <dates.src.Linear_Algebra>
    Numerical Forecast Models <dates.src.Models_Forest>
    Error Models <dates.src.Error_Models>
    Time Integration <dates.src.Time_Integration>
    Assimilation <dates.src.Assimilation>
    Utility <dates.src.Utility>
.. Visualization <dates.src.Visualization>  <-- This will be added soon!
   

Contributing to DATeS
=======================

.. toctree::
   :maxdepth: 1

    Development <dates.development>
   
   
About
======
.. toctree::
   :maxdepth: 1
   
   About (DATeS & the Team) <About>

Contact Us
===========
.. toctree::
   :maxdepth: 1
   
    Contact <Contact>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

