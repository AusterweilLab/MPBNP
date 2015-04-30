MPBNP V0.01 Release
=====

MPBNP is short for **M**assively **P**arallel **B**ayesian **N**on**p**arametric models. It is a collection of OpenCL-accelerated samplers that perform Bayesian posterior inferences for Bayesian nonparametric models (such as the Chinese Restaurant Process mixture models and the Indian Buffet Process models). It is primarily intended to make life easier for cognitive science and machine learning researchers who use Bayesian nonparametric models in their work.

MPBNP is started by the Austerweil Lab in the Department of Cognitive, Linguistic, and Psychological Sciences at Brown University. Anyone interested in this project is welcome to provide feedback and comments, or join the development!


# Installation #

To obtain MPBNP, **NEVER** use the master repository, which is almost always unstable. Instead, choose a frozen release by browsing through the different branches. The current release is v0.01.

If you have no idea what the above sentence that means, download the latest release [here](https://github.com/tqian86/MPBNP/archive/v0.01.zip). We will try to keep this link updated so that it also fetches the latest stable version of MPBNP.

To run MPBNP, please make sure that all prerequisites are installed on your computer, as described below.

## Prerequisites ##

MPBNP can be used on Windows 7/8/10, Linux and Mac OSX, with OpenCL support >= 1.1. In the following, we list the prerequisites for using MPBNP on each operating system. All following instructions apply to both desktop/workstation and laptop computers.

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

  `sudo apt-get install nvidia-346`

  and follow on-screen instructions. **A reboot is required after installation.** The graphics driver contains the OpenCL driver for nvidia graphics cards.

  > Note: In most cases, installing `nvidia-346` should enable both CUDA and OpenCL support successfully. The driver should support a range of nvidia devices, including GeForce, Quadro, and Tesla cards. However, if you have previously installed the proprietary driver *directly downloaded* from nvidia's website, chances are that `nvidia-346` won't work and you will have to stick to the [nvidia's driver](http://www.nvidia.com/Download/index.aspx?lang=en-us).

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

OpenCL drivers are usually pre-installed on all Apple computers.

> **Important!** The current Intel driver on Mac OSX contains a bug in calculating ``lgamma``, which is used in MPBNP's Chinese Restaurant Process sampler. Please avoid using the sampler on Intel HD Graphics iGPUs until a fix is released. Unfortunately, we do not know when this fix will be released on Mac OSX.

#### Python 2.7 ####

Python is pre-installed on all Apple computers. 

#### Numpy, scipy, and pyopencl ####

Both *numpy* and *scipy* are pre-installed on Apple computers upgraded to Mac OSX 10.10. To determine if *numpy* or *scipy* is installed on computers with earlier versions of Mac OSX, open a terminal window and type `python` at the command prompt. Then, try `import numpy` and `import numpy` - if any of these import commands produces an error, then the corresponding package is NOT installed on your computer. 

The recommended method for installing numpy, scipy and pyopencl is via [Macports](https://www.macports.org/). Please visit Macport's website for instructions on how to install it and obtain these packages.

You may also follow the [installation tutorial provided by PyOpenCL](http://wiki.tiker.net/PyOpenCL/Installation/Mac) if Macports looks too complicated to set up.

After these packages are installed, you should be all set.

# Usage #

Samplers in MPBNP can be invoked by executing one of the `SamplingUtility.py` program in the base directory of MPBNP. For example, to run the Chinese Restaurant Process Mixture Model sampler, look for `CRPSamplingUtility.py`. For the Indian Buffet Process sampler, look for `IBPSamplingUtility.py`.

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

> Note: despite what the help message claims, distributing chains across multiple OpenCL devices is not yet implemented. The feature is planned for future releases. 

## Chinese Restaurant Process (CRP) Sampler ##

The Chinese Restaurant Process Sampler can be invoked by running the `CRPSamplingUtility.py` program with options.

To give you an idea, here's an example:

`$ python CRPSamplingUtility --data_file ./data/normal-1d.csv --output_to_stdout --opencl`

### Prepare the input data ###

The CRP sampler expects the input data to be in a [Comma-separated Values](http://en.wikipedia.org/wiki/Comma-separated_values) (.csv) format, where each column consists of data of a single dimension. A header line is also expected. For example, the following is a perfectly fine input data set for a two-dimensional data set consisting of potentially a mixture of 2-d normal distributions:

```
V1,V2
1.1,0.1
2.2,0.3
3.3,0.2
...
```

### Choose what arguments to list ###

In this example, we perform posterior inference on class labels of data points in the `normal-1d.csv` input file, assuming that each mixture component can be represented by a normal distribution. The CRP sampler also supports category distributions (also known as a discrete distribution) as the distribution of mixture components (switching to categorical can be done by specifying `--kernel categorical`). The prior distribution on class labels is the Chinese Restaurant Process prior. We specify the input file using the argument `--data_file`, followed by the path to the input file (`normal-1d.csv` is a toy data set that comes with MPBNP). We request that the final output, i.e., the class labels of data points, be printed to standard output, which is just the screen. Finally, we use OpenCL acceleration by specifying the `--opencl` argument.

Implicit in this command is the *output mode* of the sampler, which determines how much information is recorded by the sampler. By default, the sampler is run in the *best* output mode, where the sampler discards all previous samples except for the one that produces the largest joint log probability of model and data. In other words, the *best* sample that will be recorded in the end represents the optimal tradeoff between fitness and model complexity. You can also choose to keep all samples (subject to thining and burning) that are produced in the sampling process, by specifying `--output_mode all`.

### Choose what OpenCL device to use

If you run the command above, you'll probably see something similar to the following:

```
Choose platform:
[0] <pyopencl.Platform 'Intel(R) OpenCL' at 0x39c890>
[1] <pyopencl.Platform 'AMD Accelerated Parallel Processing' at 0x7ffe9107ad30>
Choice [0]:
```

On my computer, I am asked to choose among multiple OpenCL **platforms** first because I have installed both Intel and AMD's drivers for OpenCL devices. If there is only one platform (e.g., NVIDIA's driver) installed, then you won't see this question. When there are multiple platforms, choosing a platform may lead to another set of choices:

```
Choose device(s):
[0] <pyopencl.Device 'Bonaire' on 'AMD Accelerated Parallel Processing' at 0x12a04840>
[1] <pyopencl.Device 'Intel(R) Core(TM) i3-4130T CPU @ 2.90GHz' on 'AMD Accelerated Parallel Processing' at 0x16c3da70>
Choice, comma-separated [0]:
```

Here I can select which **device** to be used for this particular sampling procedure. The first device *Bonaire* is an AMD GPU and the second device is an Intel CPU, both of which are supported by the AMD driver. Similarly, if there's only one device supported by a platform/driver, you won't see this question. 

Platforms and devices vary from computers to computers and you'll probably see something different from the example here. In general, for small-ish data sets (with data points fewer than 5K), it may actually be faster to use the CPU.

### Interpret the result ###

Continuing the example, let's suppose that I have chosen the *Bonaire* GPU to run the task. In a few seconds, the sampler will complete and print some information during and after the run. You may see something similar to this:

```
Chain 1 running, please wait ...
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199
New best sample found, loglik: -1082.82495117
New best sample found, loglik: -1078.32983398
New best sample found, loglik: -1060.10217285
New best sample found, loglik: -1017.8203125
New best sample found, loglik: -958.890075684
...
[many lines omitted]
...
New best sample found, loglik: -539.621154785
New best sample found, loglik: -539.24621582
New best sample found, loglik: -539.126953125
New best sample found, loglik: -539.058837891
New best sample found, loglik: -538.669555664
Too little improvement in loglikelihood - Abort searching
6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,1,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,1,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
Chain 1 finished. OpenCL device time: 3.860003; Total_time: 4.334000 seconds
```

This output combines both debug messages and the final output. The debug messages show the progress of the sampler, as the log probability of model and data decrease monotonically over the course of the run. The sampler intelligently stops when there is too little improvement for a sustained period of iterations. The final line of debug messages shows how much time it took to run the sampler. Here we can see the *Bonaire* GPU spent 3.86 seconds on various numerical calculations (which is not fast - the advantage of GPU computing is usually only evident when the data is really large).

The final output are the two lines consisting of numbers, one before the debug messages and one after the debug messages. The first line indexes the data points, and thus ranges from 0 to 199, covering the 200 data points in this example. The second line indexes the class label of each data point. Data points with `6` as its label are in the same cluster. The numerical differences between the names of class labels are arbitrary (i.e., a `6` data point is not more superior to a `1` data point in any sense). 

If instead of `--output_to_stdout`, the sampler was run with `--output_to_file`, then these lines representing the best sample will **not** be printed to the screen. They will be recorded in a file in the **same directory as the input file**. The screen will only display the debug messages.

Given these class labels, one can easily proceed to extract the means, variances and other statistics of interest for each class / category of data points. 

## Indian Buffet Process (IBP) Sampler ##

The Indian Buffet Process Sampler can be invoked by running the `IBPSamplingUtility.py` program with options.

To give you an idea, here's an example:

`$ python IBPSamplingUtility --data_file ./data/ibp-image.csv --output_to_stdout --opencl`

### Prepare the input data ###

The IBP sampler expects the input data to be in a [Comma-separated Values](http://en.wikipedia.org/wiki/Comma-separated_values) (.csv) format. When the noisyor likelihood function is used, each column indicates the index of a pixel, and each row indicates an individual image. Importantly, the first column of each row should indicate the actual **width** of an image (in number of pixels) and they should be identical for the same data set. The rest of the values in the csv file should consist of binary values only, which represent whether a pixel is on or off in an image. A header line is also expected. For example, the following is a perfectly fine input data set, when a noisyor likelihood function is used:

```
W,P1,P2,P3,P4
2,1,1,0,0
2,0,0,1,1
2,1,1,1,1
```

Available options are almost identical to the CRP sampler, as described in detail above. Please use the `--help` function to see supported kernels.

Note that if `--output_to_file` is selected for the tIBP sampler, it will create a directory with a name similar to the input file in the **same directory of the input file**. Inside the newly created directory, the sampler will save the sampling results in several different files. The contents of each result file should be relatively intuitive given their filenames. 

## Transformed Indian Buffet Process (tIBP) Sampler ##

The **transformed** Indian Buffet Process Sampler can be invoked by running the `tIBPSamplingUtility.py` program with options.

To give you an idea, here's an example:

`$ python tIBPSamplingUtility --data_file ./data/tibp-image-wide-n8.csv --output_to_stdout --opencl`

> MPBNP v0.01 supports two forms of transformations - vertical translation and horizontal translation. Scaling is disabled in this release due to potential problems. Scaling is planned for future releases.

### Prepare the input data ###

The tIBP sampler expects the input data to be in a [Comma-separated Values](http://en.wikipedia.org/wiki/Comma-separated_values) (.csv) format. When the noisyor likelihood function is used, each column indicates the index of a pixel, and each row indicates an individual image. Importantly, the first column of each row should indicate the actual **width** of an image (in number of pixels) and they should be identical for the same data set. The rest of the values in the csv file should consist of binary values only, which represent whether a pixel is on or off in an image. A header line is also expected. For example, the following is a perfectly fine input data set, when a noisyor likelihood function is used:

```
W,P1,P2,P3,P4
2,1,1,0,0
2,0,0,1,1
2,1,1,1,1
```

Available options are almost identical to the IBP sampler, as described in detail above. Please use the `--help` function to see supported kernels.

Note that if `--output_to_file` is selected for the tIBP sampler, it will create a directory with a name similar to the input file in the **same directory of the input file**. Inside the newly created directory, the sampler will save the sampling results in several different files. The contents of each result file should be relatively intuitive given their filenames. 

I wish to contact the author for questions, comments and suggestions.
---
Send me an email at ting_qian@brown.edu. I am a postdoc at the Austerweil Lab at Brown University. I'd love to hear from you.
