from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np
from camera import Camera
from objloader import OBJLoader

meshes = []
camera = Camera()
last_x, last_y = 0, 0
drag_mode = None
is_dragging = False
g_P = glm.mat4()

g_vertex_shader_src_lighting = '''

#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_normal; 

out vec3 vout_surface_pos;
out vec3 vout_normal;

uniform mat4 MVP;
uniform mat4 M;

void main()
{
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);
    gl_Position = MVP * p3D_in_hcoord;

    vout_surface_pos = vec3(M * vec4(vin_pos, 1));
    vout_normal = normalize( mat3(inverse(transpose(M)) ) * vin_normal);
}
'''

g_fragment_shader_src_lighting = '''
#version 330 core

in vec3 vout_surface_pos;
in vec3 vout_normal;

out vec4 FragColor;

uniform vec3 view_pos;
uniform vec3 material_color;

void main()
{
    // light and material properties
    vec3 light_pos = vec3(3,2,4);
    vec3 light_color = vec3(1,1,1);
    float material_shininess = 32.0;

    // light components
    vec3 light_ambient = 0.1*light_color;
    vec3 light_diffuse = light_color;
    vec3 light_specular = light_color;

    // material components
    vec3 material_ambient = material_color;
    vec3 material_diffuse = material_color;
    vec3 material_specular = vec3(1,1,1);  // for non-metal material

    // ambient
    vec3 ambient = light_ambient * material_ambient;

    // for diffiuse and specular
    vec3 normal = normalize(vout_normal);
    vec3 surface_pos = vout_surface_pos;
    vec3 light_dir = normalize(light_pos - surface_pos);

    // diffuse
    float diff = max(dot(normal, light_dir), 0);
    vec3 diffuse = diff * light_diffuse * material_diffuse;

    // specular
    vec3 view_dir = normalize(view_pos - surface_pos);
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow( max(dot(view_dir, reflect_dir), 0.0), material_shininess);
    vec3 specular = spec * light_specular * material_specular;

    vec3 color = ambient + diffuse + specular;
    FragColor = vec4(color, 1.);
}
'''

g_vertex_shader_src_color = '''
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

g_fragment_shader_src_color = '''
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

def drop_callback(window, paths):
    for path in paths:
        x_offset = len(meshes) * 2.0
        loader = OBJLoader(path)
        mesh = Mesh(loader.Vertices, loader.Normals, loader.Indices)
        mesh.model = glm.translate(glm.mat4(1.0), glm.vec3(x_offset, 0, 0))
        meshes.append(mesh)
        loader.print_info()

def get_view_matrix():
    return camera.get_view_matrix()

def prepare_vao_frame():
    # prepare vertex data (in main memory)
    grid_vertices = []
    for i in range(-20, 21):
                          # position   # color
        grid_vertices += [i, 0, -21, 0.7, 0.7, 0.7]
        grid_vertices += [i, 0,  21, 0.7, 0.7, 0.7]

        grid_vertices += [-21, 0, i, 0.7, 0.7, 0.7]
        grid_vertices += [ 21, 0, i, 0.7, 0.7, 0.7]
    
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

def draw_grid(vao, shader, VP, count):
    MVP_loc = glGetUniformLocation(shader, "MVP")
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(VP))
    glBindVertexArray(vao)
    glDrawArrays(GL_LINES, 0, count)

class Mesh:
    def __init__(self, vertices, normals, indices):
        self.model = glm.mat4(1.0)
        self.vertex_count = len(indices)
        self.indices = np.array(indices, dtype=np.uint32)

        interleaved = []
        for i in range(len(vertices)//3):
            interleaved += vertices[i*3:i*3+3] + normals[i*3:i*3+3]

        self.vertex_data = np.array(interleaved, dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.ebo = glGenBuffers(1)

        glBindVertexArray(self.vao)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertex_data.nbytes, self.vertex_data, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

        stride = 6 * glm.sizeof(glm.float32)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, None)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3*glm.sizeof(glm.float32)))
        glEnableVertexAttribArray(1)

    def draw(self, shader, VP):
        M = self.model
        MVP = VP * M
        glUniformMatrix4fv(glGetUniformLocation(shader, "MVP"), 1, GL_FALSE, glm.value_ptr(MVP))
        glUniformMatrix4fv(glGetUniformLocation(shader, "M"), 1, GL_FALSE, glm.value_ptr(M))
        glUniform3f(glGetUniformLocation(shader, "material_color"), 0.7, 0.7, 0.7)  # or pass color as parameter

        view_pos_loc = glGetUniformLocation(shader, 'view_pos')
        glUniform3fv(view_pos_loc, 1, glm.value_ptr(camera.get_position()))
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.vertex_count, GL_UNSIGNED_INT, None)

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
    glfwSetMouseButtonCallback(window, mouse_button_callback)
    glfwSetCursorPosCallback(window, cursor_position_callback)
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback)
    glfwSetDropCallback(window, drop_callback)
    glEnable(GL_DEPTH_TEST)

    # load shaders & get uniform locations
    shader_lighting = load_shaders(g_vertex_shader_src_lighting, g_fragment_shader_src_lighting)
    shader_color = load_shaders(g_vertex_shader_src_color, g_fragment_shader_src_color)
    
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


        # view matrix
        V = camera.get_view_matrix()
        
        glUseProgram(shader_color)
        draw_grid(vao_frame, shader_color, g_P*V, grid_vertex_count)

        glUseProgram(shader_lighting)
        for mesh in meshes:
            mesh.draw(shader_lighting, g_P*V)

        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()
