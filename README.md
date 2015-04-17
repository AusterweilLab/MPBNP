MPBNP
=====

MPBNP is short for **M**assively **P**arallel **B**ayesian **N**on**p**arametric models. It is a collection of OpenCL-accelerated samplers that perform Bayesian posterior inferences for Bayesian nonparametric models (such as the Chinese Restaurant Process mixture models and the Indian Buffet Process models). It is primarily intended to make life easier for cognitive science and machine learning researchers who use Bayesian nonparametric models in their work.

MPBNP is started by the Austerweil Lab in the Department of Cognitive, Linguistic, and Psychological Sciences at Brown University. Anyone interested in this project is welcome to provide feedback and comments, or join the development!


# Installation #

To obtain MPBNP, simply clone the git repository, or if you have no idea what that means, [download here](https://github.com/tqian86/MPBNP/archive/master.zip). To run MPBNP, please make sure that all prerequisites are installed on your computer, as described below.

## Prerequisites ##

MPBNP can be used on Windows 7/8/10, Linux and Mac OSX. In the following, we list the prerequisites for using MPBNP on each operating system. All following instructions apply to both desktop/workstation and laptop computers.

### Windows 7/8/10 (Recommended - Easy to set up) ###

#### OpenCL drivers ####

* If you have an AMD Radeon graphics card, discrete or APU, the graphics driver contains the OpenCL driver. The graphics driver should have been already installed; otherwise your display wouldn't be fully functional. Go to AMD's website for latest drivers.

  > AMD's OpenCL driver supports running OpenCL code on both their graphics cards (GPUs), and any x86 CPUs (AMD or Intel).

* If you have an nVidia graphics card, the graphics driver contains the OpenCL driver. The graphics driver should have been already installed; otherwise your display wouldn't be fully functional. Go to nvidia's website for latest drivers.

  > nVidia's OpenCL driver supports only their graphics cards.

* If you have an Intel graphics card that is integrated into an Intel CPU (known as Intel HD Graphics 4xxx and above), and do **NOT** have a discrete graphics card installed, the Intel graphics driver contains the OpenCL driver. The driver should have been already installed; otherwise your display wouldn't be fully functional. Go to Intel's website for latest drivers.

  > **Important!** The current (as of March 2015) Intel driver contains a bug in calculating ``lgamma``, which is used in MPBNP's Chinese Restaurant Process sampler. Please avoid using the sampler on Intel HD Graphics iGPUs until an updated Intel graphics driver is scheduled to release in June 2015.

#### Python 2.7 (Download the x86-64 installer from [here](https://www.python.org/downloads/release/python-279/)) ####
> When installing Python, be sure to check "Add python.exe to search path".

After installation is finished, press Windows Key + R and type `cmd` to launch a command prompt. Then type `pip install wheel` to prepare for the next step.

#### Pre-compiled [numpy](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy), [scipy](http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy), and [pyopencl](http://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl) packages for Windows maintained by Christoph Gohlke. ####

> These compiled packages have the file extension ".whl". Check for the correct version to download. This tutorial assumes you are using 64-bit Python 2.7, for which case you should download the .whl file that mentions "cp27" and "amd64" in its file name.

Install these downloaded .whl files by ``cd`` into the directory where those files are, and then type `pip install xxx.whl` (replace xxx.whl with the actual file name of a wheel file).

You should be all set.

### Ubuntu Linux 15.04 64-bit (Recommended for production purposes) ###

We strongly recommend upgrading to Ubuntu Linux version 15.04 as it comes with updated drivers for OpenCL support.

#### OpenCL drivers ####

* If you have an AMD Radeon graphics card, discrete or APU, install the AMD proprietary driver by typing the following into a terminal:

  `sudo apt-get install fglrx`
    
  and follow on-screen instructions. **A reboot is required after installation.** The graphics driver contains the OpenCL driver for AMD GPUs and x86 CPUs (both AMD and Intel).

  > To fine-tune driver version and advanced installation options, please see the instructions on this [page](https://help.ubuntu.com/community/BinaryDriverHowto/AMD). 

* If you have an nvidia graphics card, install the nvidia proprietary driver by typing the following into a terminal:

  `sudo apt-get install nvidia-current`

  and follow on-screen instructions. **A reboot is required after installation.** The graphics driver contains the OpenCL driver for nvidia graphics cards.

  > Note: if both CUDA and OpenCL need to be maintained, please pay extra attention to the dependency warnings during installation. Prior to 15.04, it is known that only the proprietary driver **directly downloaded* from nvidia's website can simultaneously support OpenCL and CUDA. However, the downloaded driver requires re-installation every time the Ubuntu kernel is updated, which is quite a hassle. Please use your own judgements for the best configuration.

* If you have an Intel graphics card that is integrated into an Intel CPU (known as Intel HD Graphics 4xxx and above), and do **NOT** have a discrete graphics card installed, the situation is a bit tricky. Intel does not provide an official OpenCL driver for Linux. However, there is an open-source project called *beignet* backed by Intel employees, which fortunately can be installed directly from Ubuntu repositories. Type

  `sudo apt-get install beignet`

  into the terminal. This should do the trick.
 
  > Beignet is known to **NOT** work on Ubuntu 14.10. Upgrading to 15.04 is necessary. Beignet is also experimental. In our experience, it sometimes hangs during execution when the same code runs just fine under Windows 7/8/10 on the same hardware (Intel provides the driver for Windows). Please proceed with caution.

#### Python 2.7, numpy, scipy, and pyopencl ####
  
* Installing these components are easy on Ubuntu, just enter the following in a terminal window:

  `sudo apt-get install python numpy scipy python-pyopencl`

You should be all set.

### Mac OSX ###

#### OpenCL drivers ####

OpenCL drivers come pre-installed on all Apple computers.

> **Important!** The current Intel driver on Mac OSX contains a bug in calculating ``lgamma``, which is used in MPBNP's Chinese Restaurant Process sampler. Please avoid using the sampler on Intel HD Graphics iGPUs until a fix is released. Unfortunately, we do not know when this fix will be released on Mac OSX.

#### Python 2.7 ####

Python comes pre-installed on all Apple computers. Open a terminal window and then type `pip install wheel` to prepare for the next step.

#### Numpy, scipy, and pyopencl ####

The recommended method for installing these packages is via [Macports](https://www.macports.org/). Please visit Macport's website for instructions on how to install it and obtain these packages.

After these packages are installed, you should be all set.

# Usage #

Samplers in MPBNP can be invoked by executing one of the `SamplingUtility.py` program in the base directory of MPBNP. For example, to run the Chinese Restaurant Process Mixture Model sampler, look for `CRPSamplingUtility.py`. For the Indian Buffet Process sampler, look for `IBPSamplingUtility.py'.

Each sampling utility follows a shared standard of parameters and arguments. There is no need to remember these parameters and arguments - when in doubt, simply run a sampling utility with the argument `--help` as in `python CRPSamplingUtility.py --help`. It will print out a detailed help message on all acceptable arguments:

```
$ python CRPSamplingUtility.py --help
usage: CRPSamplingUtility.py [-h] [--opencl] [--opencl_device {ask,gpu,cpu}]
                             --data_file DATA_FILE
                             [--kernel {gaussian,categorical}] [--iter ITER]
                             [--burnin BURNIN] [--output_mode {best,all}]
                             [--output_to_file] [--output_to_stdout]
                             [--chain CHAIN] [--distributed_chains]

A sampler for the Chinese Restaurant Process mixture model with and without
OpenCL support. Please contact Ting Qian <ting_qian@brown.edu> for questions
and feedback.

optional arguments:
  -h, --help            show this help message and exit
  --opencl              Use OpenCL acceleration
  --opencl_device {ask,gpu,cpu}
                        The device to use OpenCL acceleration on. Default
                        behavior is asking the user (i.e., you).
  --data_file DATA_FILE
  --kernel {gaussian,categorical}
                        The distribution of each component. Default is
                        gaussian/normal. Also supports categorical
                        distributions
  --iter ITER, -t ITER  The number of iterations the sampler should run. When
                        only the best sample is recorded, this parameter is
                        interpreted as the maximum number of iterations the
                        sampler will run.
  --burnin BURNIN, -b BURNIN
                        The number of iterations discarded as burn-in.
  --output_mode {best,all}
                        Output mode. Default is keeping only the sample that
                        yields the highest logliklihood of data. The other
                        option is to keep all samples.
  --output_to_file      Write posterior samples to a log file in the current
                        directory. Default behavior is not keeping records of
                        posterior samples
  --output_to_stdout    Write posterior samples to standard output (i.e., your
                        screen). Default behavior is not keeping records of
                        posterior samples
  --chain CHAIN, -c CHAIN
                        The number of chains to run. Default is 1.
  --distributed_chains  If there are multiple OpenCL devices, distribute
                        chains across them. Default is no. Will not distribute
                        to CPUs if GPU is specified in opencl_device, and vice
                        versa
```


I wish to contact the author for questions, comments and suggestions.
---
Send me an email at ting_qian@brown.edu. I am a postdoc at the Austerweil Lab at Brown University. I'd love to hear from you.
