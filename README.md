MPBNP
=====

What's MPBNP?
---
MPBNP is short for Massively Parallel Bayesian Nonparametric Models, started by the Austerweil Lab at Brown University. It is a collection of OpenCL-accelerated samplers that perform Bayesian posterior inferences for Bayesian nonparametric models (such as the Chinese Restaurant Process mixture models and the Indian Buffet Process models). It is primarily intended to make life easier for cognitive science and machine learning researchers who wish to use Bayesian nonparametric models in their work. As the toolkit currently stands, we are still a bit far from that goal. Your contribution, either in the form of feedback and comments, or in the form of joining the development, is greatly appreciated!

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