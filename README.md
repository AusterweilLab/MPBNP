MPBNP
=====

MPBNP is short for **M**assively **P**arallel **B**ayesian **N**on**p**arametric models. It is a collection of OpenCL-accelerated samplers that perform Bayesian posterior inferences for Bayesian nonparametric models (such as the Chinese Restaurant Process mixture models and the Indian Buffet Process models). It is primarily intended to make life easier for cognitive science and machine learning researchers who use Bayesian nonparametric models in their work.

MPBNP is started by the Austerweil Lab in the Department of Cognitive, Linguistic, and Psychological Sciences at Brown University. Anyone interested in this project is welcome to provide feedback and comments, or join the development!


# Installation #

## Prerequisites ##

MPBNP can be used on Windows 7/8/10, Linux and Mac OSX. In the following, we list the prerequisites for using MPBNP on each operating system.

### Windows 7/8/10 (Recommended - Easy to set up) ###

* Latest OpenCL drivers

  * If you have an AMD Radeon graphics card, discrete or APU, the graphics driver already contains the OpenCL driver. AMD's OpenCL driver supports running OpenCL code on both their graphics cards (GPUs), and any x86 CPUs (AMD or Intel).
  * If you have an nVidia graphics card, the graphics driver already contains the OpenCL driver. nVidia's OpenCL driver supports only their graphics cards.
  * If you have an Intel graphics card that is integrated into an Intel CPU (known as Intel HD Graphics 4xxx and above), and do **NOT** have a discrete graphics card installed, the Intel graphics driver already contains the OpenCL driver. 

    **Important!** The current Intel driver contains a bug in calculating ``lgamma``, which is used in MPBNP's Chinese Restaurant Process sampler. Please avoid using the sampler on Intel HD Graphics iGPUs until an updated Intel graphics driver is scheduled to release in June 2015.

* Latest 64-bit version of Python 2.7 (Download the x86-64 installer from [here](https://www.python.org/downloads/release/python-279/))

  > When installing Python, be sure to check "Add python.exe to search path".

  After installation is finished, press Windows Key + R and type "cmd" to launch a command prompt. Then type `pip install wheel` to prepare for the next step.

* Latest pre-compiled [numpy](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy), [scipy](http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy), and [pyopencl](http://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl) packages for Windows maintained by Christoph Gohlke.

  > These compiled packages have the file extension ".whl". Check for the correct version to download. This tutorial assumes you are using 64-bit Python 2.7, for which case you should download the .whl file that mentions "cp27" and "amd64" in its file name.

  Install these downloaded .whl files by ``cd`` into the directory where those files are, and then type ``pip install xxx.whl`` (replace xxx.whl with the actual file name of a wheel file).

You are all set.

What's "OpenCL-accelerated"?
---
Performing inferences for Bayesian nonparametric models can be quite slow due to the sheer number of calculations needed. MPBNP samplers implement the core inference procedures in OpenCL, which accelerates the computational process by utilizing multi-core CPUs and GPUs. Besides the OpenCL parts, MPBNP is  written in Python 2 so that you don't have to be a programming guru to understand what it does.

Cool. How faster is it?
---
You can do your own benchmarking using the tools that come with the package. In our experience, the core computation part (time spent on writing the data to the hard disk, for example, is not considered) can be 50X faster than a C-based sampler using the same algorithm. If you typically code up your sampler in Python or R, the speedup will be even more significant.

I don't know much about programming. Just show me how to use it.
---
Sure. We will provide a tutorial shortly. But if you have on your operating system:

* Python 2.7 or greater (No Python 3 support yet)
* pyopencl 2013.2 or greater
* AMD, nvidia, or Intel driver for OpenCL support (depending on what hardware you have)

then you should be able to run the samplers right out of the box. 

Don't worry if this seems overwhelming - we will write detailed tutorials on how to obtain these software programs soon!

I wish to contact the author for questions, comments and suggestions.
---
Send me an email at ting_qian@brown.edu. I am a postdoc at the Austerweil Lab at Brown University. I'd love to hear from you.
