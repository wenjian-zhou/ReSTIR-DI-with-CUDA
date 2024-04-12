cmake_minimum_required(VERSION 3.0)

project(ReSTIRDI CXX CUDA)

include_directories( ${CMAKE_SOURCE_DIR}/include)
link_directories( ${CMAKE_SOURCE_DIR}/lib)

add_executable(ReSTIRDI
    cutil_math.h
    ReSTIRDI.h 
    ReSTIRDI.cu
    main.cpp
    ./include/ImGui/imconfig.h
	
	./include/ImGui/imgui.h
    ./include/ImGui/imconfig.h
    ./include/ImGui/imgui_impl_glut.h
    ./include/ImGui/imgui_impl_opengl3.h 
    ./include/ImGui/imgui_impl_opengl3_loader.h
    ./include/ImGui/imgui_internal.h 
    ./include/ImGui/imstb_rectpack.h 
    ./include/ImGui/imstb_textedit.h 
    ./include/ImGui/imstb_truetype.h
    
    ./include/ImGui/imgui.cpp 
    ./include/ImGui/imgui_demo.cpp 
    ./include/ImGui/imgui_draw.cpp 
    ./include/ImGui/imgui_impl_glut.cpp
    ./include/ImGui/imgui_impl_opengl3.cpp 
    ./include/ImGui/imgui_tables.cpp 
    ./include/ImGui/imgui_widgets.cpp )

target_link_libraries(ReSTIRDI
    freeglut
    freeglut_static
    glew32
    glew32s)

find_package(OpenGL REQUIRED)