cmake_minimum_required(VERSION 3.0)

project(ReSTIRDI CXX CUDA)

include_directories( ${CMAKE_SOURCE_DIR}/include)
link_directories( ${CMAKE_SOURCE_DIR}/lib)

add_executable(ReSTIRDI
    cutil_math.h
    ReSTIRDI.h 
    ReSTIRDI.cu
    main.cpp)

target_link_libraries(ReSTIRDI
    freeglut
    freeglut_static
    glew32
    glew32s)

find_package(OpenGL REQUIRED)