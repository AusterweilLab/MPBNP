MPBNP
=====

MPBNP is short for **M**assively **P**arallel **B**ayesian **N**on**p**arametric models. It is a collection of OpenCL-accelerated samplers that perform Bayesian posterior inferences for Bayesian nonparametric models (such as the Chinese Restaurant Process mixture models and the Indian Buffet Process models). It is primarily intended to make life easier for cognitive science and machine learning researchers who use Bayesian nonparametric models in their work.

MPBNP is started by the Austerweil Lab in the Department of Cognitive, Linguistic, and Psychological Sciences at Brown University. Anyone interested in this project is welcome to provide feedback and comments, or join the development!


# Installation #

## Prerequisites ##

MPBNP can be used on Windows 7/8/10, Linux and Mac OSX. In the following, we list the prerequisites for using MPBNP on each operating system.

### Windows 7/8/10 (Recommended - Easy to set up) ###

* Latest OpenCL drivers

  * If you have an AMD Radeon graphics card, discrete or APU, the graphics driver (which should have been already installed; otherwise your display wouldn't be fully functional) contains the OpenCL driver. 
  > AMD's OpenCL driver supports running OpenCL code on both their graphics cards (GPUs), and any x86 CPUs (AMD or Intel).

  * If you have an nVidia graphics card, the graphics driver (which should have been already installed; otherwise your display wouldn't be fully functional) contains the OpenCL driver. 
  > nVidia's OpenCL driver supports only their graphics cards.

  * If you have an Intel graphics card that is integrated into an Intel CPU (known as Intel HD Graphics 4xxx and above), and do **NOT** have a discrete graphics card installed, the Intel graphics driver (which should have been already installed; otherwise your display wouldn't be fully functional) contains the OpenCL driver. 
  <font color="red"> **Important!** The current (as of March 2015) Intel driver contains a bug in calculating ``lgamma``, which is used in MPBNP's Chinese Restaurant Process sampler. Please avoid using the sampler on Intel HD Graphics iGPUs until an updated Intel graphics driver is scheduled to release in June 2015.</font>

* Latest 64-bit version of Python 2.7 (Download the x86-64 installer from [here](https://www.python.org/downloads/release/python-279/))

  > When installing Python, be sure to check "Add python.exe to search path".

  After installation is finished, press Windows Key + R and type `cmd` to launch a command prompt. Then type `pip install wheel` to prepare for the next step.

* Latest pre-compiled [numpy](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy), [scipy](http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy), and [pyopencl](http://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl) packages for Windows maintained by Christoph Gohlke.

  > These compiled packages have the file extension ".whl". Check for the correct version to download. This tutorial assumes you are using 64-bit Python 2.7, for which case you should download the .whl file that mentions "cp27" and "amd64" in its file name.

  Install these downloaded .whl files by ``cd`` into the directory where those files are, and then type `pip install xxx.whl` (replace xxx.whl with the actual file name of a wheel file).

You should be all set.

### Ubuntu Linux 15.04 64-bit ###

We recommend upgrading to Ubuntu Linux version 15.04 as it comes with updated drivers for OpenCL support.


I wish to contact the author for questions, comments and suggestions.
---
Send me an email at ting_qian@brown.edu. I am a postdoc at the Austerweil Lab at Brown University. I'd love to hear from you.
