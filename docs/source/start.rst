Getting Started
===============

Installation
************
We recommend installing miniconda from - https://conda.io/projects/conda/en/latest/user-guide/install/index.html if you are new to python. 
 

Create new virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Setup new virtual environment with an environment file that installs dependencies while creating environment::

    conda env create –f environment.yml 
    conda activate <title>

**OR**

Install to existing environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install with pip/conda to an existing/no environment (after uploading our library to pypi/conda-forge)::

    pip/conda install <title>

**OR**

From source
^^^^^^^^^^^
Download library as zip or clone with ‘git’ and use the functions
 

Other optional setup 
^^^^^^^^^^^^^^^^^^^^
- Pytorch
- Spleeter (source separation) 


Usage
*****
Once installed you can start using the library in one of the following ways. See our tutorials for more comprehensive examples. 

1. Open anaconda prompt on Windows, or the regular terminal on Mac and Linux 

2. Start the virtual environment if you created one::

    conda activate <title>

3. Start a python shell and import the library::

    python3 
    import <title>

4. Or start jupyter notebook, open a new/existing notebook in the web browser interface and import the library (see one of the tutorial notebooks for an example)::

    jupyter-notebook 
    import <title>


Tutorials 
*********
See below example jupyter notebook files to help you get started:

1. Load audio, compute pitch track, compute spectrogram, plot all (`Notebook 1 <https://github.com/rohitma38/rtd-test-code-2/blob/main/example%20notebooks/example-1.ipynb>`_)
2. Load audio, compute onsets, compute tempogram, plot all (`Notebook 2 <https://github.com/rohitma38/rtd-test-code-2/blob/main/example%20notebooks/example-tempogram.ipynb>`_)
