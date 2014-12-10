#!/usr/bin/env python

import pyopencl as cl
import pyopencl.array
import numpy
from time import time

class CL:
    def __init__(self):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

    def loadProgram(self, filename):
        #read in the OpenCL source file as a string
        f = open(filename, 'r')
        fstr = "".join(f.readlines())
        #print fstr
        #create the program
        self.program = cl.Program(self.ctx, fstr).build()

    def popCorn(self):
        mf = cl.mem_flags

        #initialize client side (CPU) arrays
        numpy.random.seed(10)
        self.a = numpy.random.random((2048,64)).astype(numpy.float32)
        self.b = numpy.random.random((64,2048)).astype(numpy.float32)

        #create OpenCL buffers
        self.a_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.a)
        self.b_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.b)
        self.dest_buf = cl.array.empty(self.queue, (self.a.shape[0], self.b.shape[1]), numpy.float32)

    def execute(self):
        self.program.dotProd(self.queue, self.dest_buf.shape, None,
                             self.dest_buf.data, self.a_buf, self.b_buf, 
                             numpy.int32(self.a.shape[1]))
        #self.program.dotProd2(self.queue, (self.dest_buf.shape[0],), None,
        #                      self.dest_buf.data, self.a_buf, self.b_buf, 
        #                      numpy.int32(self.a.shape[1]), numpy.int32(self.b.shape[1]))
        c = self.dest_buf.get()
        return c



if __name__ == "__main__":
    example = CL()
    example.loadProgram("../kernels/utilities_cl.c")
    example.popCorn()
    a_time = time()
    #print(example.a, example.b)
    result = example.execute()
    print('GPU time:', time() - a_time)
    #print(result)

    a_time = time()
    cpu_result = numpy.dot(example.a, example.b)
    print('CPU time:', time() - a_time)
    #print(cpu_result)
    print((result - cpu_result).sum())
