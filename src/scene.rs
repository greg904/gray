use crate::light::Light;
use crate::sphere::Sphere;
use crate::triangle::Triangle;
use crate::Camera;

use glam::Vec3;

pub struct Material {
    pub albedo: Vec3,
}

pub struct TriangleObject {
    pub tri: Triangle,
    pub mat: Material,
}

pub struct SphereObject {
    pub sph: Sphere,
    pub mat: Material,
}

pub struct Scene {
    pub spheres: Vec<SphereObject>,
    pub triangles: Vec<TriangleObject>,
    pub lights: Vec<Light>,
    pub camera: Camera,
}
