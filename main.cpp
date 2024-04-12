#include "ReSTIRDI.h"
#include <ImGui/imgui.h>
#include <ImGui/imgui_impl_glut.h>
#include <ImGui/imgui_impl_opengl3.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
//#include <GL/GL.h>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <cuda_gl_interop.h>

GLuint vbo;
float3* finalOutputBuffer = NULL;
int frames = 0;
Reservoir* previousReservoir = NULL;
Reservoir* currentReservoir = NULL;
bool moveCamera = false;
bool useReSTIR = false;
bool temporalReuse = false;
bool spatialReuse = false;

void CreateCudaAndCpuMemory(){
	// allocate CUDA memory
	cudaMalloc((void**)&previousReservoir, scr_width * scr_height * sizeof(Reservoir));
	cudaMalloc((void**)&currentReservoir, scr_width * scr_height * sizeof(Reservoir));
}

void DeleteCudaAndCpuMemory(){
	// free CUDA memory
	cudaFree(finalOutputBuffer);
	cudaFree(previousReservoir);
	cudaFree(currentReservoir);
}

void CreateVBO(GLuint* vbo)
{
	//Create vertex buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	//Initialize VBO
	unsigned int size = scr_width * scr_height * sizeof(float3);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//Register VBO with CUDA
	cudaGLRegisterBufferObject(*vbo);
}

void Display(void) {
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGLUT_NewFrame();
	ImGui::NewFrame();
	ImGuiIO& io = ImGui::GetIO();

	ImGui::Begin("ReSTIR DI");
	ImGui::Text("ReSTIR DI control panel:");
	ImGui::Checkbox("Use ReSTIR DI", &useReSTIR);
	ImGui::Checkbox("Temporal Reuse", &temporalReuse);
	ImGui::Checkbox("Spatial Reuse", &spatialReuse);
	ImGui::End();

	ImGui::Render();

	frames++;
    cudaDeviceSynchronize();
    cudaGLMapBufferObject((void**)&finalOutputBuffer, vbo);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	const int current_t = glutGet(GLUT_ELAPSED_TIME);

	RenderGate(finalOutputBuffer, frames, WangHash(frames), 
				previousReservoir, currentReservoir, 
				useReSTIR, temporalReuse, spatialReuse);

    cudaDeviceSynchronize();
	cudaGLUnmapBufferObject(vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(2, GL_FLOAT, 12, 0);
	glColorPointer(4, GL_UNSIGNED_BYTE, 12, (GLvoid*)8);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glDrawArrays(GL_POINTS, 0, scr_width * scr_height);
	glDisableClientState(GL_VERTEX_ARRAY);

	glUseProgram(0);
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	glutSwapBuffers();
}

void Idle()
{
	glutPostRedisplay(); //CS6610 Requirement
}


int main(int argc, char** argv) {
    glutInitContextVersion(4, 5);
	glutInitContextProfile(GLUT_COMPATIBILITY_PROFILE);
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(scr_width, scr_height);
	glutInitWindowPosition(0, 0);
	glutCreateWindow("ReSTIR DI");

	int cuda_devices[1];
	unsigned int num_cuda_devices;
	cudaGLGetDevices(&num_cuda_devices, cuda_devices, 1, cudaGLDeviceListAll);
    cudaSetDevice(cuda_devices[0]);

    glDisable(GL_DEPTH_TEST);

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glMatrixMode(GL_PROJECTION);
	gluOrtho2D(0.0, scr_width, 0.0, scr_height);

    glutDisplayFunc(Display);
    glutIdleFunc(Idle);

	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;       // Enable Keyboard Controls

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();

	// Setup Platform/Renderer bindings
	ImGui_ImplGLUT_Init();
	ImGui_ImplOpenGL3_Init();

	ImGui_ImplGLUT_InstallFuncs();

	// call glewInit() after creating the OpenGL window
	glewInit();

    CreateVBO(&vbo);

	CreateCudaAndCpuMemory();
	//produceReference();

    glutMainLoop();

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGLUT_Shutdown();
	ImGui::DestroyContext();

    DeleteCudaAndCpuMemory();

	return 0;
}