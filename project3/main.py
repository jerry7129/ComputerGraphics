from OpenGL.GL import *
from glfw.GLFW import *
import glm
import math
import os
import ctypes
import numpy as np
import time

class OBJLoader:
    def __init__(self, path):
        self.Vertices = []
        self.Normals = []
        self.Indices = []
        self.positions = []
        self.normals = []
        self.faces = []
        self.vertex_dict = {}
        self.next_index = 0
        self.path = path
        self.load(path)

    def load(self, path):
        with open(path, 'r') as file:
            for line in file:
                if line.startswith('v '):
                    self.positions.append(list(map(float, line.strip().split()[1:])))
                elif line.startswith('vn '):
                    self.normals.append(list(map(float, line.strip().split()[1:])))
                elif line.startswith('f '):
                    face = []
                    for part in line.strip().split()[1:]:
                        vals = part.split('/')
                        v_idx = int(vals[0])
                        n_idx = int(vals[2]) if len(vals) >= 3 and vals[2] else 0
                        face.append((v_idx, n_idx))
                    self.faces.append(face)
        self.triangulate()

    def get_or_add_vertex(self, v_idx, n_idx):
        key = (v_idx, n_idx)
        if key in self.vertex_dict:
            return self.vertex_dict[key]
        self.Vertices.extend(self.positions[v_idx - 1])
        self.Normals.extend(self.normals[n_idx - 1])
        self.vertex_dict[key] = self.next_index
        self.next_index += 1
        return self.vertex_dict[key]

    def triangulate(self):
        for face in self.faces:
            if len(face) < 3:
                continue
            for i in range(1, len(face) - 1):
                tri = [face[0], face[i], face[i + 1]]
                for v_idx, n_idx in tri:
                    idx = self.get_or_add_vertex(v_idx, n_idx)
                    self.Indices.append(idx)

    def print_info(self):
        print(f"OBJ file name: {os.path.basename(self.path)}")
        print(f"Total Faces: {len(self.faces)}")
        tri = sum(1 for f in self.faces if len(f) == 3)
        quad = sum(1 for f in self.faces if len(f) == 4)
        ngons = sum(1 for f in self.faces if len(f) > 4)
        print(f"Triangles: {tri}, Quads: {quad}, N-gons: {ngons}")

class Camera:
    def __init__(self, center=glm.vec3(0, 0, 0), radius=3.0, azimuth=0.0, elevation=0.5):
        self.center = center
        self.radius = radius
        self.azimuth = azimuth
        self.elevation = elevation

        self.min_elevation = -math.pi / 2 + 0.01
        self.max_elevation = math.pi / 2 - 0.01

        self.up = glm.vec3(0, 1, 0)

    def get_position(self):
        x = self.radius * math.cos(self.elevation) * math.sin(self.azimuth)
        y = self.radius * math.sin(self.elevation)
        z = self.radius * math.cos(self.elevation) * math.cos(self.azimuth)
        return self.center + glm.vec3(x, y, z)

    def get_view_matrix(self):
        return glm.lookAt(self.get_position(), self.center, self.up)

    def orbit(self, dx, dy):
        self.azimuth += dx * 0.005
        self.elevation += dy * 0.005
        self.elevation = max(min(self.elevation, self.max_elevation), self.min_elevation)

    def pan(self, dx, dy):
        right = glm.normalize(glm.cross(self.get_position() - self.center, self.up))
        up = glm.normalize(glm.cross(right, self.get_position() - self.center))
        self.center += right * dx * 0.005 + up * dy * 0.005

    def zoom(self, dy):
        self.radius *= 1.0 + dy * 0.01

class BVHNode:
    def __init__(self, name):
        self.name = name
        self.offset = glm.vec3(0)
        self.channels = []
        self.children = []
        self.parent = None
        self.box = None

class BVHLoader:
    def __init__(self, path):
        self.path = path
        self.root = None
        self.frames = []
        self.frame_time = 0
        self.joint_names = []
        with open(path, 'r') as f:
            self.lines = [line.strip() for line in f if line.strip()]
        self.index = 0
        self.root = self.parse_hierarchy()
        self.parse_motion()

    def parse_hierarchy(self):
        def parse_node():
            line = self.lines[self.index]
            tokens = line.split()

            if tokens[0] == "End":
                self.index += 1  # Skip "End Site"
                self.index += 1  # Skip "{"
                while not self.lines[self.index].startswith("}"):
                    self.index += 1
                self.index += 1  # Skip "}"
                return None

            if tokens[0] not in ["ROOT", "JOINT"]:
                print(f"Error: unexpected token at line {self.index}: {tokens}")
                return None

            name = tokens[1]
            node = BVHNode(name)
            self.joint_names.append(name)

            self.index += 1  # Skip line with ROOT or JOINT
            if self.lines[self.index] != "{":
                print(f"Error: expected '{{' after {tokens[0]} {name}")
                return None
            self.index += 1  # Skip "{"

            if not self.lines[self.index].startswith("OFFSET"):
                print(f"Error: expected OFFSET at line {self.index}")
                return None
            offset_tokens = self.lines[self.index].split()
            node.offset = glm.vec3(*map(float, offset_tokens[1:]))
            self.index += 1

            if not self.lines[self.index].startswith("CHANNELS"):
                print(f"Error: expected CHANNELS at line {self.index}")
                return None
            channel_tokens = self.lines[self.index].split()
            node.channels = channel_tokens[1:]
            self.index += 1

            while not self.lines[self.index].startswith("}"):
                if self.lines[self.index].startswith("JOINT") or self.lines[self.index].startswith("End"):
                    child = parse_node()
                    if child:
                        child.parent = node
                        node.children.append(child)
                else:
                    self.index += 1
            self.index += 1  # Skip "}"
            return node

        while not self.lines[self.index].startswith("ROOT"):
            self.index += 1
        return parse_node()

    def parse_motion(self):
        while not self.lines[self.index].startswith("MOTION"):
            self.index += 1
        self.index += 1
        num_frames = int(self.lines[self.index].split()[1])
        self.index += 1
        self.frame_time = float(self.lines[self.index].split()[2])
        self.index += 1
        for _ in range(num_frames):
            frame = list(map(float, self.lines[self.index].split()))
            self.frames.append(frame)
            self.index += 1
        print("Loaded BVH:", os.path.basename(self.path))
        print("Frames:", num_frames)
        print("FPS:", round(1/self.frame_time, 2))
        print("Joints:", len(self.joint_names))
        print("Joint Names:", self.joint_names)

def create_unit_box():
    # 박스 원형 생성 (y축 기준 길이 1.0짜리 박스)
    vertices = [
        -0.05, 0.0, -0.05,
         0.05, 0.0, -0.05,
         0.05, 0.0,  0.05,
        -0.05, 0.0,  0.05,
        -0.05, 1.0, -0.05,
         0.05, 1.0, -0.05,
         0.05, 1.0,  0.05,
        -0.05, 1.0,  0.05,
    ]
    indices = [
        0,1,2, 2,3,0,
        4,5,6, 6,7,4,
        0,1,5, 5,4,0,
        2,3,7, 7,6,2,
        0,3,7, 7,4,0,
        1,2,6, 6,5,1
    ]
    normals = [0, 1, 0] * 8
    return Mesh(vertices, normals, indices)

def draw_bvh_rest_pose(node, parent_transform, VP, shader):
    if node is None:
        return

    # 현재 노드의 로컬 변환 (offset만 반영)
    local_transform = glm.translate(glm.mat4(1), node.offset)
    global_transform = parent_transform * local_transform

    # 현재 노드의 위치
    parent_pos = glm.vec3(parent_transform[3])  # 부모 기준 위치
    current_pos = glm.vec3(global_transform[3])  # 현재 노드 위치

    if node.parent:
        mesh = create_bone(parent_pos, current_pos)
        if mesh:
            mesh.draw(shader, VP)

    for child in node.children:
        draw_bvh_rest_pose(child, global_transform, VP, shader)

def create_bone(p1, p2):
    direction = p2 - p1
    length = glm.length(direction)
    if length < 1e-6:
        return None

    direction = glm.normalize(direction)
    up = glm.vec3(0, 1, 0)
    axis = glm.cross(up, direction)
    angle = math.acos(glm.dot(up, direction))

    # 정렬: 단위 큐브를 y축 방향으로 만들어놓은 뒤, 회전 + 스케일 + 위치이동
    model = glm.mat4(1)
    model = glm.translate(model, p1)
    if glm.length(axis) > 1e-6:
        model = glm.rotate(model, angle, axis)
    THICKNESS = 0.5
    model = glm.scale(model, glm.vec3(THICKNESS, length, THICKNESS))

    mesh = create_unit_box()
    mesh.model = model
    return mesh

def draw_bvh_frame(node, parent_matrix, frame_data, channel_offset, VP, shader):
    if node is None:
        return channel_offset

    T = glm.mat4(1.0)

    # # root 위치 채널 처리
    if node.parent is None:
        x, y, z = frame_data[channel_offset:channel_offset+3] 
        T = glm.translate(T, glm.vec3(x, y, z))
        channel_offset += 3
    else:
        T = glm.translate(T, node.offset)

    rotation_order = []
    for ch in node.channels:
        if ch.endswith("rotation"):
            rotation_order.append(ch)

    # apply rotations in order
    for ch in rotation_order:
        angle = math.radians(frame_data[channel_offset])
        if ch.startswith("X"):
            T = T * glm.rotate(glm.mat4(1), angle, glm.vec3(1, 0, 0))
        elif ch.startswith("Y"):
            T = T * glm.rotate(glm.mat4(1), angle, glm.vec3(0, 1, 0))
        elif ch.startswith("Z"):
            T = T * glm.rotate(glm.mat4(1), angle, glm.vec3(0, 0, 1))
        channel_offset += 1

    global_transform = parent_matrix * T

    # 좌표 계산
    parent_pos = glm.vec3(parent_matrix[3])
    current_pos = glm.vec3(global_transform[3])

    if node.parent:
        direction = current_pos - parent_pos
        length = glm.length(direction)
        if length > 1e-6:
            direction = glm.normalize(direction)
            up = glm.vec3(0, 1, 0)
            axis = glm.cross(up, direction)
            angle = math.acos(glm.dot(up, direction))

            # 정렬: 단위 큐브를 y축 방향으로 만들어놓은 뒤, 회전 + 스케일 + 위치이동
            model = glm.mat4(1)
            model = glm.translate(model, parent_pos)
            if glm.length(axis) > 1e-6:
                model = glm.rotate(model, angle, axis)
            model = glm.scale(model, glm.vec3(1, length, 1))

            mesh = create_unit_box()
            mesh.model = model
            if mesh:
                # mesh.model = global_transform
                mesh.draw(shader, VP, color=(0.7, 0.7, 0.7))

    for child in node.children:
        channel_offset = draw_bvh_frame(child, global_transform, frame_data, channel_offset, VP, shader)

    return channel_offset

use_obj_mode = False
obj_bvh_loader = None
obj_mesh_map = {}

def load_obj_bvh():
    global obj_bvh_loader, obj_mesh_map, use_obj_mode
    obj_bvh_loader = BVHLoader(os.path.join("data", "test.bvh"))
    obj_mesh_map.clear()
    for joint_name in obj_bvh_loader.joint_names:
        obj_path = os.path.join("data", f"{joint_name}.obj")
        if not os.path.exists(obj_path):
            print(f"Missing OBJ for joint: {joint_name}")
            continue
        loader = OBJLoader(obj_path)
        mesh = Mesh(loader.Vertices, loader.Normals, loader.Indices)
        obj_mesh_map[joint_name] = mesh
    use_obj_mode = True

def draw_obj_frame(node, parent_matrix, frame_data, channel_offset, VP, shader):
    if node is None:
        return channel_offset

    T = glm.mat4(1.0)
    if node.parent is None:
        x, y, z = frame_data[channel_offset:channel_offset+3]
        T = glm.translate(T, glm.vec3(x, y, z))
        channel_offset += 3
    else:
        T = glm.translate(T, node.offset)

    rotation_order = [ch for ch in node.channels if ch.endswith("rotation")]
    for ch in rotation_order:
        angle = math.radians(frame_data[channel_offset])
        axis = glm.vec3(1, 0, 0) if ch.startswith("X") else glm.vec3(0, 1, 0) if ch.startswith("Y") else glm.vec3(0, 0, 1)
        T *= glm.rotate(glm.mat4(1), angle, axis)
        channel_offset += 1

    global_transform = parent_matrix * T
    if node.name in obj_mesh_map:
        mesh = obj_mesh_map[node.name]
        
        center = compute_obj_center(mesh)
        center_transform = glm.translate(glm.mat4(1.0), -center)

        mesh.model = global_transform * center_transform
        mesh.draw(shader, VP, color=(0.2, 0.6, 1.0))

    for child in node.children:
        channel_offset = draw_obj_frame(child, global_transform, frame_data, channel_offset, VP, shader)

    return channel_offset

def compute_obj_center(mesh):
    vertices = mesh.vertex_data.reshape(-1, 6)[:, :3]
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    center = (min_coords + max_coords) / 2.0
    return glm.vec3(*center)

meshes = []
camera = Camera()
last_x, last_y = 0, 0
drag_mode = None
is_dragging = False
g_P = glm.mat4()

bvh_loader = None
bvh_meshes = []
is_animating = False
last_time = 0
current_frame = 0

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
    global bvh_loader, bvh_meshes, is_animating, current_frame
    for path in paths:
        if path.endswith('.bvh'):
            bvh_loader = BVHLoader(path)
            is_animating = False
            current_frame = 0
        elif path.endswith('.obj'):
            x_offset = len(meshes) * 2.0
            loader = OBJLoader(path)
            mesh = Mesh(loader.Vertices, loader.Normals, loader.Indices)
            mesh.model = glm.translate(glm.mat4(1.0), glm.vec3(x_offset, 0, 0))
            meshes.append(mesh)
            loader.print_info()
            
def key_callback(window, key, scancode, action, mods):
    global is_animating, last_time
    if action == GLFW_PRESS:
        if key == GLFW_KEY_SPACE:
            is_animating = not is_animating
            last_time = time.time()
        elif key == GLFW_KEY_1:
            load_obj_bvh()

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

    def draw(self, shader, VP, color=(0.7, 0.7, 0.7)):
        M = self.model
        MVP = VP * M
        glUniformMatrix4fv(glGetUniformLocation(shader, "MVP"), 1, GL_FALSE, glm.value_ptr(MVP))
        glUniformMatrix4fv(glGetUniformLocation(shader, "M"), 1, GL_FALSE, glm.value_ptr(M))
        glUniform3f(glGetUniformLocation(shader, "material_color"), *color)  # or pass color as parameter


        view_pos_loc = glGetUniformLocation(shader, 'view_pos')
        glUniform3fv(view_pos_loc, 1, glm.value_ptr(camera.get_position()))
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.vertex_count, GL_UNSIGNED_INT, None)

def main():
    global g_P, last_time, current_frame
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
    glfwSetKeyCallback(window, key_callback)
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

        if bvh_loader:
            if is_animating:
                now = time.time()
                if now - last_time >= bvh_loader.frame_time:
                    current_frame = (current_frame + 1) % len(bvh_loader.frames)
                    last_time = now
                draw_bvh_frame(bvh_loader.root, glm.mat4(1), bvh_loader.frames[current_frame], 0, g_P * V, shader_lighting)
            else:
                draw_bvh_rest_pose(bvh_loader.root, glm.mat4(1), g_P * V, shader_lighting)

        if use_obj_mode and obj_bvh_loader:
            if is_animating:
                now = time.time()
                if now - last_time >= obj_bvh_loader.frame_time:
                    current_frame = (current_frame + 1) % len(obj_bvh_loader.frames)
                    last_time = now
                draw_obj_frame(obj_bvh_loader.root, glm.mat4(1), obj_bvh_loader.frames[current_frame], 0, g_P * V, shader_lighting)
            else:
                draw_obj_frame(obj_bvh_loader.root, glm.mat4(1), [0]*len(obj_bvh_loader.frames[0]), 0, g_P * V, shader_lighting)
        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()
