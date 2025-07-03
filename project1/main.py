from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np
from camera import Camera

camera = Camera()
last_x, last_y = 0, 0
drag_mode = None
is_dragging = False
g_P = glm.mat4()

g_vertex_shader_src = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_color; 

out vec4 vout_color;

uniform mat4 MVP;

void main()
{
    // 3D points in homogeneous coordinates
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);

    gl_Position = MVP * p3D_in_hcoord;

    vout_color = vec4(vin_color, 1.);
}
'''

g_fragment_shader_src = '''
#version 330 core

in vec4 vout_color;

out vec4 FragColor;

void main()
{
    FragColor = vout_color;
}
'''

def load_shaders(vertex_shader_source, fragment_shader_source):
    # build and compile our shader program
    # ------------------------------------
    
    # vertex shader 
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)    # create an empty shader object
    glShaderSource(vertex_shader, vertex_shader_source) # provide shader source code
    glCompileShader(vertex_shader)                      # compile the shader object
    
    # check for shader compile errors
    success = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(vertex_shader)
        print("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" + infoLog.decode())
        
    # fragment shader
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)    # create an empty shader object
    glShaderSource(fragment_shader, fragment_shader_source) # provide shader source code
    glCompileShader(fragment_shader)                        # compile the shader object
    
    # check for shader compile errors
    success = glGetShaderiv(fragment_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(fragment_shader)
        print("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" + infoLog.decode())

    # link shaders
    shader_program = glCreateProgram()               # create an empty program object
    glAttachShader(shader_program, vertex_shader)    # attach the shader objects to the program object
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)                    # link the program object

    # check for linking errors
    success = glGetProgramiv(shader_program, GL_LINK_STATUS)
    if (not success):
        infoLog = glGetProgramInfoLog(shader_program)
        print("ERROR::SHADER::PROGRAM::LINKING_FAILED\n" + infoLog.decode())
        
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return shader_program    # return the shader program

def mouse_button_callback(window, button, action, mods):
    global is_dragging, drag_mode

    if button == GLFW_MOUSE_BUTTON_LEFT:
        if action == GLFW_PRESS:
            if mods & GLFW_MOD_ALT:
                is_dragging = True
                if mods & GLFW_MOD_SHIFT:
                    drag_mode = 'pan'
                elif mods & GLFW_MOD_CONTROL:
                    drag_mode = 'zoom'
                else:
                    drag_mode = 'orbit'
        elif action == GLFW_RELEASE:
            is_dragging = False
            drag_mode = None

def cursor_position_callback(window, xpos, ypos):
    global last_x, last_y, is_dragging, drag_mode

    alt_down = glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS
    shift_down = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS
    ctrl_down = glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS

    if is_dragging:
        if drag_mode == 'orbit' and not alt_down:
            is_dragging = False
            drag_mode = None
            return
        elif drag_mode == 'pan' and not (alt_down and shift_down):
            is_dragging = False
            drag_mode = None
            return
        elif drag_mode == 'zoom' and not (alt_down and ctrl_down):
            is_dragging = False
            drag_mode = None
            return

    if not is_dragging:
        last_x, last_y = xpos, ypos
        return

    dx = xpos - last_x
    dy = ypos - last_y
    last_x, last_y = xpos, ypos

    if drag_mode == 'orbit':
        camera.orbit(dx, dy)
    elif drag_mode == 'pan':
        camera.pan(dx, dy)
    elif drag_mode == 'zoom':
        camera.zoom(dy)

def framebuffer_size_callback(window, width, height):
    global g_P

    glViewport(0, 0, width, height)

    perse_height = 10.
    perse_width = perse_height * width/height
    g_P = glm.perspective(45, perse_width/perse_height, 0.1, 100)


def get_view_matrix():
    return camera.get_view_matrix()

def prepare_vao_frame():
    # prepare vertex data (in main memory)
    grid_vertices = []
    for i in range(-20, 21):
                          # position   # color
        grid_vertices += [i/10, 0, -2.1, 0.7, 0.7, 0.7]
        grid_vertices += [i/10, 0,  2.1, 0.7, 0.7, 0.7]

        grid_vertices += [-2.1, 0, i/10, 0.7, 0.7, 0.7]
        grid_vertices += [ 2.1, 0, i/10, 0.7, 0.7, 0.7]
    
    vertices = np.array(grid_vertices, np.float32)

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO, len(vertices)


def main():
    global g_P
    # initialize glfw
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)   # OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)  # Do not allow legacy OpenGl API calls
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE) # for macOS

    # create a window and OpenGL context
    window = glfwCreateWindow(800, 800, '2022083409', None, None)
    if not window:
        glfwTerminate()
        return
    glfwMakeContextCurrent(window)

    # register event callbacks
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback)

    # load shaders
    shader_program = load_shaders(g_vertex_shader_src, g_fragment_shader_src)

    # get uniform locations
    MVP_loc = glGetUniformLocation(shader_program, 'MVP')
    
    # prepare vaos
    vao_frame, grid_vertex_count = prepare_vao_frame()

    # initialize projection matrix
    perse_height = 10.
    perse_width = perse_height * 800/800    # initial width/height
    g_P = glm.perspective(45, perse_width/perse_height, 0.1, 100)

    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        # render

        # enable depth test (we'll see details later)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        glUseProgram(shader_program)


        # projection matrix
        # use orthogonal projection (we'll see details later)
        # P = glm.perspective(glm.radians(45.0), 1.0, 0.1, 100.0)

        # view matrix
        V = camera.get_view_matrix()
        
        # current frame: P*V*I (now this is the world frame)
        I = glm.mat4()
        MVP = g_P*V*I
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))

        # draw current frame
        glBindVertexArray(vao_frame)
        glDrawArrays(GL_LINES, 0, grid_vertex_count)

        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()
