1. Installation
***************

Prerequisites
=============
| Make sure you have the following:  
| 1. **Configured SAP HANA** (skip this step if you have an instance with PAL enabled). 
| There are 2 ways to do that.  
| In HANA Cloud:
* `Create <https://www.sap.com/cmp/td/sap-hana-cloud-trial.html>`_ a free trial account  
* `Setup <https://saphanajourney.com/hana-cloud/learning-article/how-to-create-your-trial-sap-hana-cloud-instance/>`_ an instance  
* `Enable <https://help.sap.com/viewer/db19c7071e5f4101837e23f06e576495/2020_03_QRC/en-US/502e458a260d4445810e6b9094c5b7e7.html>`_ PAL - Predictive Analysis Library. It is vital to enable it because we use their algorithms.

| In Virtual Machine:
* Rent a virtual machine in Azure, AWS, Google Cloud, etc.
* `Install <https://developers.sap.com/group.hxe-install-binary.html>`_ HANA instance there or on your PC (if you have >32 Gb RAM).
* `Enable <https://www.youtube.com/watch?v=NyEIj76aqFg&list=PLkzo92owKnVy6nOZMFZIZxcvBCoRdshsR&index=19>`_ PAL - Predictive Analysis Library. It is vital to enable it because we use their algorithms.

Install
=======

1. Make sure you have `Python 3.6 - Python 3.9 <https://www.python.org/downloads/>`_ installed
2. Install `Cython <https://pypi.org/project/Cython/>`_::
        
        pip3 install Cython

3. Install the library from `pypi <https://pypi.org>`_ (stable)::
    
        pip3 install hana_automl
   
   or `GitHub repository <https://github.com/dan0nchik/SAP-HANA-AutoML>`_ (latest version)::

        pip3 install https://github.com/dan0nchik/SAP-HANA-AutoML/archive/dev.zip

.. Caution:: GitHub version is not stable, it may contain bugs!

Available on all operating systems where Python can be installed :)