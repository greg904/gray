use crate::light::Light;
use crate::scene::Scene;
use crate::scene::SphereObject;
use crate::scene::TriangleObject;
use crate::sphere::Sphere;
use crate::spherical;
use crate::triangle::Triangle;
use crate::Camera;
use crate::Ray;
use crate::BACKGROUND;

use std::f32;

use glam::Vec2;
use glam::Vec3A;

use rand::rngs::SmallRng;

pub struct RayTracer<'a> {
    pub scene: &'a Scene,
    pub samples: usize,
    pub bounces: u8,
    pub aim_rays: bool,
    pub use_avg_diffuse: bool,
}

enum HitObjectInfo<'a> {
    Sphere(&'a SphereObject),
    Triangle {
        triangle: &'a TriangleObject,
        uv: Vec2,
        xyz: Vec3A,
    },
    Nothing,
}

struct Hit<'a> {
    t: f32,
    obj_info: HitObjectInfo<'a>,
}

impl<'a> Hit<'a> {
    fn intersection_normal_albedo(&self, ray: &Ray) -> (Vec3A, Vec3A, Vec3A) {
        match self.obj_info {
            HitObjectInfo::Sphere(sphere) => {
                let intersection = ray.point_from_t(self.t);
                let normal = (intersection - *sphere.sph.center()).normalize();
                (intersection, normal, sphere.mat.albedo)
            }
            HitObjectInfo::Triangle {
                triangle,
                xyz,
                uv: _uv,
            } => (xyz, triangle.tri.normal(), triangle.mat.albedo),
            HitObjectInfo::Nothing => unreachable!(),
        }
    }
}

impl<'a> RayTracer<'a> {
    fn first_intersection_with_ray(&self, ray: &Ray) -> Hit {
        let mut first = Hit {
            t: f32::NAN,
            obj_info: HitObjectInfo::Nothing,
        };

        for sphere in self.scene.spheres.iter() {
            let t = sphere.sph.t_of_intersection_point_with_ray(ray);
            // Operations with `NaN` return `false`.
            if t.is_nan() || t >= first.t {
                continue;
            }
            first = Hit {
                t,
                obj_info: HitObjectInfo::Sphere(sphere),
            };
        }

        for triangle in self.scene.triangles.iter() {
            let t = triangle.tri.t_of_intersection_point_with_ray_checked(ray);
            // The follow comparison evaluates to `true` if `t` is `NaN` (triangle plane and ray
            // are parallel or ray is not seeing the visible face of the triangle).
            if !(t >= 0.) {
                continue;
            }
            if t >= first.t {
                continue;
            }
            let xyz = ray.point_from_t(t);
            let uv = triangle.tri.uv_of_point(xyz);
            if !Triangle::uv_is_inside(uv) {
                continue;
            }
            first = Hit {
                t,
                obj_info: HitObjectInfo::Triangle { triangle, uv, xyz },
            };
        }

        first
    }

    fn compute_direct_lighting_for_light(&self, point: Vec3A, normal: Vec3A, light: &Light) -> Vec3A {
        let point_to_light = *light.pos() - point;
        let d_cos_theta = normal.dot(point_to_light);
        if d_cos_theta <= 0. {
            return Vec3A::ZERO;
        }
        let d = point_to_light.length();

        // Check if something is hiding the light.
        let ray = Ray {
            dir: point_to_light / d,
            origin: point,
        };
        for sphere in self.scene.spheres.iter() {
            let t = sphere.sph.t_of_intersection_point_with_ray(&ray);
            // t could be `NaN`, in which case the following comparison will return `false`.
            if t <= d {
                return Vec3A::ZERO;
            }
        }
        for triangle in self.scene.triangles.iter() {
            let t = triangle.tri.t_of_intersection_point_with_ray_checked(&ray);
            // The comparison is done with a negation so that it evaluates to `true` for `NaN`.
            if !(t <= d) {
                continue;
            }
            let xyz = ray.point_from_t(t);
            let uv = triangle.tri.uv_of_point(xyz);
            if !Triangle::uv_is_inside(uv) {
                continue;
            }
            return Vec3A::ZERO;
        }

        let cos_theta = d_cos_theta / d;
        let point_light_factor_times_4pi = cos_theta / (d * d);
        point_light_factor_times_4pi * *light.power_over_4pi()
    }

    pub fn compute_direct_lighting(&self, point: Vec3A, normal: Vec3A) -> Vec3A {
        self.scene.lights.iter().fold(Vec3A::ZERO, |acc, l| {
            acc + self.compute_direct_lighting_for_light(point, normal, l)
        })
    }

    pub fn compute_indirect_lighting(
        &self,
        point: Vec3A,
        normal: Vec3A,
        rng: &mut SmallRng,
        recursion_level: u8,
    ) -> Vec3A {
        if recursion_level >= self.bounces {
            return Vec3A::ZERO;
        }

        let (dir, cos_theta) = crate::random_unit_vector_in_hemisphere_and_cos_theta(normal, rng);
        let next_ray = Ray {
            origin: point,
            dir,
        };
        self.ray_trace_indirect(&next_ray, rng, recursion_level + 1) * cos_theta
    }

    fn ray_trace_indirect(&self, ray: &Ray, rng: &mut SmallRng, recursion_level: u8) -> Vec3A {
        let closest_hit = self.first_intersection_with_ray(ray);
        if closest_hit.t.is_nan() {
            return Vec3A::new(BACKGROUND.0, BACKGROUND.1, BACKGROUND.2);
        };
        let (intersection, normal, albedo) = closest_hit.intersection_normal_albedo(ray);
        let direct_lighting = self.compute_direct_lighting(intersection, normal);
        let indirect_lighting =
            self.compute_indirect_lighting(intersection, normal, rng, recursion_level);

        albedo * (direct_lighting + indirect_lighting)
    }

    pub fn ray_trace(&self, ray: &Ray, rng: &mut SmallRng) -> Vec3A {
        let closest_hit = self.first_intersection_with_ray(ray);
        if closest_hit.t.is_nan() {
            return Vec3A::new(BACKGROUND.0, BACKGROUND.1, BACKGROUND.2);
        };
        let (intersection, normal, albedo) = closest_hit.intersection_normal_albedo(ray);
        let direct_lighting = self.compute_direct_lighting(intersection, normal);

        let mut indirect_lighting = Vec3A::ZERO;

        if self.aim_rays {
            let mut area_by_index: [(usize, f32); 10] = [(usize::MAX, 0.); 10];
            let mut cursor = 0;
            let mut total_area = 0.;

            let mut indirect_lighting_area = 0.;

            for (i, sphere) in self.scene.spheres.iter().enumerate() {
                // The sphere is visible from our side.
                if normal.dot(*sphere.sph.center() - intersection) <= 0.00001 {
                    continue;
                }
                let d = sphere.sph.center().distance(intersection) as f32;
                let r = sphere.sph.radius() as f32;
                let area = spherical::area_of_sphere_projected_to_unit_sphere(d, r);
                if area <= 0.00001 {
                    continue;
                }
                area_by_index[cursor] = (i, area);
                cursor += 1;
                total_area += area;
            }

            for (i, triangle) in self.scene.triangles.iter().enumerate() {
                // The triangle is visible from our side.
                if triangle.tri.normal().dot(triangle.tri.a() - intersection) >= -0.00001
                    && triangle.tri.normal().dot(triangle.tri.b() - intersection) >= -0.00001
                    && triangle.tri.normal().dot(triangle.tri.c() - intersection) >= -0.00001
                {
                    continue;
                }
                let a_ = (triangle.tri.a() - intersection).normalize();
                let b_ = (triangle.tri.b() - intersection).normalize();
                let c_ = (triangle.tri.c() - intersection).normalize();
                let area =
                    spherical::area_of_intersection_of_spherical_triangle_and_unit_hemisphere(
                        a_, b_, c_, normal,
                    );
                if area <= 0.00001 {
                    continue;
                }
                area_by_index[cursor] = (self.scene.spheres.len() + i, area);
                cursor += 1;
                total_area += area;
            }

            if cursor > 0 {
                for i in 0..self.samples {
                    let j = i % cursor;
                    let samples_per_j = 1 + (self.samples - j - 1) / cursor;
                    let obj = area_by_index[j].0;

                    let (next_intersection, next_ray_dir, next_normal, next_albedo) =
                        if obj < self.scene.spheres.len() {
                            let sphere = &self.scene.spheres[obj];
                            let target = spherical::random_look_in_sphere(
                                intersection,
                                *sphere.sph.center(),
                                sphere.sph.radius(),
                                rng,
                            );
                            let normal = (target - *sphere.sph.center()).normalize();
                            (
                                target,
                                (target - intersection).normalize(),
                                normal,
                                sphere.mat.albedo,
                            )
                        } else {
                            let triangle = &self.scene.triangles[obj - self.scene.spheres.len()];
                            let dir = spherical::random_direction_toward_triangle(
                                triangle.tri.a() - intersection,
                                triangle.tri.b() - intersection,
                                triangle.tri.c() - intersection,
                                rng,
                            );
                            let ray = Ray {
                                origin: intersection,
                                dir,
                            };
                            let t = triangle.tri.t_of_intersection_point_with_ray(&ray);
                            let pt = ray.point_from_t(t);
                            (pt, dir, triangle.tri.normal(), triangle.mat.albedo)
                        };

                    let area = area_by_index[j].1;
                    let sample_probability = 1. / samples_per_j as f32;

                    let cos_theta = next_ray_dir.dot(normal) as f32;
                    if cos_theta > 0. {
                        let area_cos_theta = area * cos_theta;
                        let diffuse = if self.use_avg_diffuse && area_cos_theta < 0.4 {
                            self.scene.primitives_avg_diffuse[obj]
                        } else {
                            let next_direct_lighting =
                                self.compute_direct_lighting(next_intersection, next_normal);
                            let next_indirect_lighting = self.compute_indirect_lighting(
                                next_intersection,
                                next_normal,
                                rng,
                                2,
                            );
                            next_albedo * (next_direct_lighting + next_indirect_lighting)
                        };

                        indirect_lighting += diffuse * area_cos_theta * sample_probability;
                    }

                    indirect_lighting_area += area * sample_probability;
                }
            }

            let ambient_area = 2. * f32::consts::PI - total_area;
            if ambient_area > 0. {
                let indirect_lighting_cos_theta_factor = 0.5;
                indirect_lighting += Vec3A::new(BACKGROUND.0, BACKGROUND.1, BACKGROUND.2)
                    * indirect_lighting_cos_theta_factor
                    * ambient_area;
                indirect_lighting_area += ambient_area;
            }

            indirect_lighting /= indirect_lighting_area;
        } else {
            for _ in 0..self.samples {
                indirect_lighting += self.compute_indirect_lighting(intersection, normal, rng, 1);
            }
            indirect_lighting /= self.samples as f32;
        }

        albedo * (direct_lighting + indirect_lighting)
    }
}
