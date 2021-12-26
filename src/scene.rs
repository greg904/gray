use crate::light::Light;
use crate::ray_trace::RayTracer;
use crate::sphere::Sphere;
use crate::triangle::Triangle;
use crate::Camera;

use glam::Vec2;
use glam::Vec3;

use rand::distributions::Distribution;
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::UnitSphere;

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
    pub primitives_avg_diffuse: Vec<Vec3>,
    pub lights: Vec<Light>,
    pub camera: Camera,
}

impl Scene {
    pub fn compute_primitives_avg_diffuse(&mut self) {
        // The values we are computng globally affect a frame. If we were to
        // use a non-deterministic RNG, it would make some frames brighter than
        // others resulting in flickering.
        let mut rng = SmallRng::seed_from_u64(0);

        for i in 0..self.spheres.len() {
            let sphere = &self.spheres[i];
            let samples = 1;
            let mut total_diffuse = Vec3::ZERO;
            for _ in 0..samples {
                let offset: [f32; 3] = UnitSphere.sample(&mut rng);
                let pt = Vec3::new(
                    sphere.sph.center().x + sphere.sph.radius() * offset[0],
                    sphere.sph.center().y + sphere.sph.radius() * offset[1],
                    sphere.sph.center().z + sphere.sph.radius() * offset[2],
                );
                let rt = RayTracer {
                    aim_rays: false,
                    bounces: 2,
                    samples: 0,
                    use_avg_diffuse: false,
                    scene: &self,
                };
                let normal = (pt - *sphere.sph.center()).normalize();
                let direct = rt.compute_direct_lighting(pt, normal);
                let indirect = rt.compute_indirect_lighting(pt, normal, &mut rng, 1);
                let diffuse = sphere.mat.albedo * (direct + indirect);
                total_diffuse += diffuse;
            }
            self.primitives_avg_diffuse[i] = total_diffuse / samples as f32;
        }

        for i in 0..self.triangles.len() {
            let triangle = &self.triangles[i];
            let samples = 1;
            let mut total_diffuse = Vec3::ZERO;
            for _ in 0..samples {
                let uv: Vec2 = rng.gen();
                let pt = triangle.tri.point_from_uv(uv);
                let rt = RayTracer {
                    aim_rays: false,
                    bounces: 2,
                    samples: 0,
                    use_avg_diffuse: false,
                    scene: &self,
                };
                let direct = rt.compute_direct_lighting(pt, triangle.tri.normal());
                let indirect = rt.compute_indirect_lighting(pt, triangle.tri.normal(), &mut rng, 1);
                let diffuse = triangle.mat.albedo * (direct + indirect);
                total_diffuse += diffuse;
            }
            self.primitives_avg_diffuse[self.spheres.len() + i] = total_diffuse / samples as f32;
        }
    }
}
