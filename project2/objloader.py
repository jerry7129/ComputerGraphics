import os

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
