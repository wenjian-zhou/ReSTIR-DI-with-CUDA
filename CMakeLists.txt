cmake_minimum_required(VERSION 3.0)

project(ReSTIRDI CXX CUDA)

add_executable(ReSTIRDI
    cutil_math.h
    ReSTIRDI.h 
    ReSTIRDI.cu)

