import glm
import math

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
