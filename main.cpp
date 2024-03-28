#include "ReSTIRDI.h"
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GL/GL.h>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <cuda_gl_interop.h>

GLuint vbo;
float3* finaloutputbuffer = NULL;

void deleteCudaAndCpuMemory(){
	// free CUDA memory
	cudaFree(finaloutputbuffer);
}

void createVBO(GLuint* vbo)
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

void disp(void) {
    cudaThreadSynchronize();
    cudaGLMapBufferObject((void**)&finaloutputbuffer, vbo);

    glClear(GL_COLOR_BUFFER_BIT);

    render_gate(finaloutputbuffer);
    //printf("%f\n", *finaloutputbuffer[3].x);

    cudaThreadSynchronize();
	cudaGLUnmapBufferObject(vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(2, GL_FLOAT, 12, 0);
	glColorPointer(4, GL_UNSIGNED_BYTE, 12, (GLvoid*)8);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glDrawArrays(GL_POINTS, 0, scr_width * scr_height);
	glDisableClientState(GL_VERTEX_ARRAY);
    //glDisableClientState(GL_COLOR_ARRAY);
	glutSwapBuffers();
}

void Idle()
{
	glutPostRedisplay(); //CS6610 Requirement
}

void Keys(unsigned char key, int x, int y){
    switch (key)
	{
	case 27: glutLeaveMainLoop(); //Requirement5: press esc to exit
		break;
	default:
		break;
	}
}
void SpecialKeys(int key, int x, int y) {}
void Mouse(int button, int state, int x, int y) {}
void MouseMotion(int x, int y) {}


int main(int argc, char** argv) {
    glutInitContextVersion(4, 5);
	glutInit(&argc, argv); //to initialize GLUT
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(scr_width, scr_height); //Requirement3: Specifying the size of the window
	glutInitWindowPosition(100, 100);
	glutCreateWindow("ReSTIR DI");

    cudaGLSetGLDevice(0);
    cudaSetDevice(0);

    //glDisable(GL_DEPTH_TEST);

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glMatrixMode(GL_PROJECTION);
	gluOrtho2D(0.0, scr_width, 0.0, scr_height);

    glutDisplayFunc(disp);
    glutKeyboardFunc(Keys);
    glutSpecialFunc(SpecialKeys);
    glutMouseFunc(Mouse);
    glutMotionFunc(MouseMotion);
    glutIdleFunc(Idle);

    // call glewInit() after creating the OpenGL window
	glewInit();

    createVBO(&vbo);

    glutMainLoop();

    deleteCudaAndCpuMemory();
	return 0;
}