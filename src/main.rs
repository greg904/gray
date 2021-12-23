use core::f32;
use core::time::Duration;

use std::sync::Arc;
use std::sync::Condvar;
use std::sync::Mutex;
use std::sync::RwLock;
use std::thread;

use glam::DMat2;
use glam::DVec2;
use glam::DVec3;
use glam::Mat3;
use glam::Vec3;

use minifb::Window;
use minifb::WindowOptions;

use rand::distributions::Distribution;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::UnitSphere;

const WIDTH: usize = 1080;
const HEIGHT: usize = 720;
const SAMPLES: usize = 4;
const WORKER_COUNT: usize = 8;

struct Ray {
    origin: DVec3,
    dir: DVec3,
}

fn random_unit_vector_in_hemisphere(center: DVec3, rng: &mut SmallRng) -> DVec3 {
    let v = UnitSphere.sample(rng);
    let mut v = DVec3::new(v[0], v[1], v[2]);
    // Make sure that the vector is in the hemisphere.
    if v.dot(center) < 0. {
        v = -v;
    }
    v
}

impl Ray {
    fn from_origin_and_random_direction_in_hemisphere(
        origin: DVec3,
        center: DVec3,
        rng: &mut SmallRng,
    ) -> Self {
        Self {
            origin,
            dir: random_unit_vector_in_hemisphere(center, rng),
        }
    }

    fn point_from_t(&self, t: f64) -> DVec3 {
        self.origin + t * self.dir
    }
}

struct Material {
    albedo: Vec3,
}

struct Triangle {
    points: (DVec3, DVec3, DVec3),
    normal: DVec3,
    s_xyz: DVec3,
    t_xyz: DVec3,
    uv_to_st: DMat2,
    mat: Material,
}

impl Triangle {
    fn new(points: (DVec3, DVec3, DVec3), mat: Material) -> Self {
        // We will need to find the UV coordinates of any point in the plane that contains the
        // triangle quickly. One way to do that is to create an orthonormal basis for the
        // points in the plane, because it will then be easy to get the
        // coordinates of our point in that base, unlike with the non orthnonormal basis formed
        // from the triangle's vertices), and then to multiply it by a change of basis matrix to
        // get the UV coordinates.

        // 1. Create our orthonormal basis.
        let s_xyz = (points.1 - points.0).normalize();
        let mut t_xyz = points.2 - points.0;
        t_xyz -= t_xyz.dot(s_xyz) * s_xyz;
        t_xyz = t_xyz.normalize();

        // 2. Create the change of basis matrix.
        let v_xyz = points.2 - points.0;
        let u_st = DVec2::new((points.1 - points.0).length(), 0f64);
        let v_st = DVec2::new(v_xyz.dot(s_xyz), v_xyz.dot(t_xyz));
        let st_to_uv = DMat2::from_cols(u_st, v_st);
        let uv_to_st = st_to_uv.inverse();

        Triangle {
            points,
            normal: (points.1 - points.0).cross(points.2 - points.0).normalize(),
            s_xyz,
            t_xyz,
            uv_to_st,
            mat,
        }
    }

    /// Returns the `t` parameter of the point of intersection between the ray and the plane that
    /// contains the triangle, or `NaN` if there is none. Note that although the intersection
    /// point is in the plane that contains the triangle, it could still be outside the triangle.
    fn t_of_intersection_point_with_ray(&self, ray: &Ray) -> f64 {
        -((ray.origin - self.points.0).dot(self.normal)) / ray.dir.dot(self.normal)
    }

    /// Returns the UV coordinates of a point in space, assuming that the point is in the plane
    /// that contains the triangle.
    fn uv_of_point(&self, point_xyz: DVec3) -> DVec2 {
        // Get the coordinates of the point in the orthonormal basis.
        let point_st = DVec2::new(
            (point_xyz - self.points.0).dot(self.s_xyz),
            (point_xyz - self.points.0).dot(self.t_xyz),
        );
        // Use the change of basis matrix.
        self.uv_to_st * point_st
    }

    /// Returns whether the given UV coordinates are inside the triangle or not.
    fn uv_is_inside(uv: DVec2) -> bool {
        uv.x >= 0. && uv.y >= 0. && uv.x + uv.y <= 1.
    }

    /// Converts UV coordinates to world coordinates.
    fn point_from_uv(&self, uv: DVec2) -> DVec3 {
        self.points.0
            + uv.x * (self.points.1 - self.points.0)
            + uv.y * (self.points.2 - self.points.0)
    }
}

struct Sphere {
    center: DVec3,
    radius: f64,
    radius_sqr: f64,
    mat: Material,
}

impl Sphere {
    fn new(center: DVec3, radius: f64, mat: Material) -> Self {
        Self {
            center,
            radius,
            radius_sqr: radius * radius,
            mat,
        }
    }

    fn t_of_intersection_point_with_ray(&self, ray: &Ray) -> Option<f64> {
        // Let $O(x_O,y_O,z_O)$ be the center of the sphere, $R(x_R,y_R,z_R)$ the starting point of the ray and $\vec{u}(x_u,y_u,z_u)$ a unit vector for the direction of the ray.
        //
        // A point $M(x,y,z)$ on the ray is described by its parameter $t \in \mathbb{R}_+$ by the following equations : $x = x_R + tx_u,$ $y = y_R + ty_u,$ and $z = z_R + tz_u.$
        //
        // The equation of the sphere is: $(x - x_O)^2 + (y - y_O)^2 + (z - z_O)^2 = r^2.$
        //
        // By combining the two equations, we find the points of intersection between the ray and the sphere:
        // \begin{gather*}
        // (x_R + tx_u - x_O)^2 + (y_R + ty_u - y_O)^2 + (z_R + tz_u - z_O)^2 = r^2\\
        // (tx_u + (x_R - x_O))^2 + (ty_u + (y_R - y_O))^2 + (tz_u + (z_R - z_O))^2 = r^2
        // \end{gather*}
        // \begin{align*}
        // &t^2x_u^2 + 2tx_u(x_R - x_O) + (x_R - x_O)^2\\
        // 	+ &t^2y_u^2 + 2ty_u(y_R - y_O) + (y_R - y_O)^2\\
        // 	+ &t^2z_u^2 + 2tz_u(z_R - z_O) + (z_R - z_O)^2 = r^2
        // \end{align*}
        // \begin{align*}
        // &(x_u^2 + y_u^2 + z_u^2)t^2\\
        // + &(2x_u(x_R - x_O) + 2y_u(y_R - y_O) + 2z_u(z_R - z_O))t\\
        // + &(x_R - x_O)^2 + (y_R - y_O)^2 + (z_R - z_O)^2 - r^2 = 0
        // \end{align*}
        //
        // This is a quadratic equation which we can solve.
        let a = ray.dir.length_squared();
        let b = 2. * ray.dir.dot(ray.origin - self.center);
        let b_sqr = b * b;
        let c = (ray.origin - self.center).length_squared() - self.radius_sqr;
        let discriminant = b_sqr - 4. * a * c;
        if discriminant < 0. {
            return None;
        }
        let discriminant_sqrt = discriminant.sqrt();
        // If both `t_0` and `t_1` are negative, the sphere is behind the ray.
        // If one is positive and the other is negative, the ray origin is inside the sphere.
        // In that case, we do not want to count the intersection because our spheres are only
        // made up of the exterior surface. This allows shooting a ray from a point on the
        // sphere without caring about intersecting that same sphere again without convoluted
        // tricks.
        let t_0 = (-b - discriminant_sqrt) / (2. * a);
        if t_0 < 0. {
            return None;
        }
        let t_1 = (-b + discriminant_sqrt) / (2. * a);
        if t_1 < 0. {
            return None;
        }
        // Take the first point on the ray, that is the point with the smallest parameter.
        let t = t_0.min(t_1);
        Some(t)
    }
}

struct Light {
    pos: DVec3,
    power: Vec3,
}

struct Camera {
    center: DVec3,
    dir: DVec3,
    z_near: f64,
    vertical_fov: f64,
    // Precomputed values.
    grid_center: DVec3,
    grid_right: DVec3,
    grid_up: DVec3,
}

impl Camera {
    fn new(center: DVec3, dir: DVec3, z_near: f64, vertical_fov: f64) -> Self {
        let grid_center = center + z_near * dir;
        let grid_scale = 2. * (vertical_fov / 2.).tan() * z_near;
        let grid_right = grid_scale * dir.cross(DVec3::Z).normalize();
        let grid_up = grid_scale * grid_right.cross(dir).normalize();
        Self {
            center,
            dir,
            z_near,
            vertical_fov,
            grid_center,
            grid_right,
            grid_up,
        }
    }

    /// Returns the ray that needs to be shot to give the color for a pixel.
    ///
    /// `y` must be between `-1` and `1` inclusive, while `x` is in the
    /// corresponding range such that the aspect ratio is respected. For
    /// example, with an aspect ratio of `2:1`, then `x` lies between
    /// `-2` and `2` inclusive.
    fn ray_for_pixel(&self, x: f64, y: f64) -> Ray {
        let pixel = self.grid_center + x * self.grid_right + y * self.grid_up;
        Ray {
            origin: pixel,
            dir: (pixel - self.center).normalize(),
        }
    }
}

struct Scene {
    spheres: Vec<Sphere>,
    triangles: Vec<Triangle>,
    lights: Vec<Light>,
    camera: Camera,
}

enum HitObjectInfo<'a> {
    Sphere(&'a Sphere),
    Triangle {
        triangle: &'a Triangle,
        uv: DVec2,
        xyz: DVec3,
    },
}

struct Hit<'a> {
    t: f64,
    obj_info: HitObjectInfo<'a>,
}

impl<'a> Hit<'a> {
    fn intersection_normal_albedo(&self, ray: &Ray) -> (DVec3, DVec3, Vec3) {
        match self.obj_info {
            HitObjectInfo::Sphere(sphere) => {
                let intersection = ray.point_from_t(self.t);
                let normal = (intersection - sphere.center).normalize();
                (intersection, normal, sphere.mat.albedo)
            }
            HitObjectInfo::Triangle {
                triangle,
                xyz,
                uv: _uv,
            } => (xyz, triangle.normal, triangle.mat.albedo),
        }
    }
}

impl Scene {
    fn first_intersection_with_ray(&self, ray: &Ray) -> Option<Hit> {
        let mut first: Option<Hit> = None;

        for sphere in self.spheres.iter() {
            let t = match sphere.t_of_intersection_point_with_ray(ray) {
                Some(val) => val,
                None => continue,
            };
            if first.as_ref().map(|h| t >= h.t).unwrap_or(false) {
                continue;
            }
            first = Some(Hit {
                obj_info: HitObjectInfo::Sphere(sphere),
                t,
            });
        }

        for triangle in self.triangles.iter() {
            let t = triangle.t_of_intersection_point_with_ray(ray);
            if t.is_nan() || t < 0. {
                continue;
            }
            if first.as_ref().map(|h| t >= h.t).unwrap_or(false) {
                continue;
            }
            if ray.dir.dot(triangle.normal) >= 0. {
                // The triangle can only be seen when its normal is pointing
                // toward us.
                continue;
            }
            let xyz = ray.point_from_t(t);
            let uv = triangle.uv_of_point(xyz);
            if !Triangle::uv_is_inside(uv) {
                continue;
            }
            first = Some(Hit {
                obj_info: HitObjectInfo::Triangle { triangle, uv, xyz },
                t,
            });
        }

        first
    }

    fn compute_direct_lighting(&self, point: DVec3, normal: DVec3) -> Vec3 {
        let mut total = Vec3::ZERO;

        'light_iter: for light in self.lights.iter() {
            let point_to_light = light.pos - point;

            // Check if something is hiding the light.
            let ray = Ray {
                dir: point_to_light.normalize(),
                origin: point,
            };
            let light_t = point_to_light.length();
            for sphere in self.spheres.iter() {
                if let Some(t) = sphere.t_of_intersection_point_with_ray(&ray) {
                    if t <= light_t {
                        continue 'light_iter;
                    }
                }
            }
            for triangle in self.triangles.iter() {
                let t = triangle.t_of_intersection_point_with_ray(&ray);
                if t.is_nan() || t > light_t {
                    continue;
                }
                if ray.dir.dot(triangle.normal) >= 0. {
                    // The triangle can only be seen when its normal is pointing
                    // toward us.
                    continue;
                }
                let xyz = ray.point_from_t(t);
                let uv = triangle.uv_of_point(xyz);
                if !Triangle::uv_is_inside(uv) {
                    continue;
                }
                continue 'light_iter;
            }

            let cos_theta = normal.dot(ray.dir) as f32;
            let d = light_t as f32;
            let point_light_factor = cos_theta.max(0.) / (4. * f32::consts::PI * d * d);
            total += point_light_factor * light.power;
        }

        total
    }

    fn ray_trace_indirect(&self, ray: &Ray, rng: &mut SmallRng, recursion_level: u8) -> Vec3 {
        if recursion_level > 2 {
            return Vec3::ZERO;
        }

        let closest_hit = match self.first_intersection_with_ray(ray) {
            Some(val) => val,
            None => return Vec3::ZERO,
        };
        let (intersection, normal, albedo) = closest_hit.intersection_normal_albedo(ray);
        let direct_lighting = self.compute_direct_lighting(intersection, normal);
        let indirect_lighting = self.ray_trace_indirect(
            &Ray::from_origin_and_random_direction_in_hemisphere(intersection, normal, rng),
            rng,
            recursion_level + 1,
        );

        albedo * (direct_lighting + indirect_lighting)
    }

    fn ray_trace(&self, ray: &Ray, rng: &mut SmallRng) -> Vec3 {
        let closest_hit = match self.first_intersection_with_ray(ray) {
            Some(val) => val,
            None => return Vec3::ZERO,
        };
        let (intersection, normal, albedo) = closest_hit.intersection_normal_albedo(ray);
        let direct_lighting = self.compute_direct_lighting(intersection, normal);

        let mut indirect_lighting = Vec3::ZERO;
        for _ in 0..SAMPLES {
            indirect_lighting += self.ray_trace_indirect(
                &Ray::from_origin_and_random_direction_in_hemisphere(intersection, normal, rng),
                rng,
                1,
            );
        }
        indirect_lighting /= SAMPLES as f32;

        albedo * (direct_lighting + indirect_lighting)
    }
}

fn f32_to_u8(f: f32) -> u8 {
    (f * 255.).clamp(0., 255.) as u8
}

fn gamma_encode(c: Vec3) -> Vec3 {
    c.powf(0.45)
}

fn rrt_and_odf_fit(v: Vec3) -> Vec3 {
    // From https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl (MIT licensed).
    let a = v * (v + 0.0245786f32) - 0.000090537f32;
    let b = v * (0.983729f32 * v + 0.4329510f32) + 0.238081f32;
    a / b
}

fn aces_fit(color: Vec3) -> Vec3 {
    // From https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl (MIT licensed).
    let input_mat = Mat3::from_cols_array(&[
        0.59719f32, 0.07600f32, 0.02840f32, 0.35458f32, 0.90834f32, 0.13383f32, 0.04823f32,
        0.01566f32, 0.83777f32,
    ]);
    let color = input_mat * color;
    let color = rrt_and_odf_fit(color);
    let output_mat = Mat3::from_cols_array(&[
        1.60475f32,
        -0.10208f32,
        -0.00327f32,
        -0.53108f32,
        1.10813f32,
        -0.07276f32,
        -0.07367f32,
        -0.00605f32,
        1.07602f32,
    ]);
    let color = output_mat * color;
    color
}

fn main() {
    let mut buf: Vec<u32> = vec![0; WIDTH * HEIGHT];
    let mut window = Window::new("gray", WIDTH, HEIGHT, WindowOptions::default())
        .expect("failed to create window");

    let scene = Arc::new(RwLock::new(Scene {
        spheres: vec![Sphere::new(
            DVec3::ZERO,
            0.5,
            Material {
                albedo: Vec3::new(1., 0., 0.),
            },
        )],
        triangles: vec![
            Triangle::new(
                (
                    DVec3::new(-50., -50., -0.7),
                    DVec3::new(50., -50., -0.7),
                    DVec3::new(-50., 50., -0.7),
                ),
                Material {
                    albedo: Vec3::new(1., 1., 1.),
                },
            ),
            Triangle::new(
                (
                    DVec3::new(50., 50., -0.7),
                    DVec3::new(-50., 50., -0.7),
                    DVec3::new(50., -50., -0.7),
                ),
                Material {
                    albedo: Vec3::new(1., 1., 1.),
                },
            ),
        ],
        lights: vec![Light {
            pos: DVec3::new(10., 17., 50.),
            power: Vec3::new(40000., 40000., 40000.),
        }],
        camera: Camera::new(
            DVec3::new(-1., 0., 2.),
            DVec3::new(1., 0., -1.).normalize(),
            0.1,
            80f64.to_radians(),
        ),
    }));

    let mut theta = 0f32;

    let total_pixels = WIDTH * HEIGHT;
    let pixels_per_worker = total_pixels / WORKER_COUNT;
    let mut worker_bufs = Vec::new();
    let mut worker_handles = Vec::new();
    let mut all_running = Vec::new();
    for i in 0..WORKER_COUNT {
        let pixels = if i == WORKER_COUNT {
            // The last worker gets the remaining workers as the division was rounded down.
            total_pixels - (WORKER_COUNT - 1) * pixels_per_worker
        } else {
            pixels_per_worker
        };
        let buf = Arc::new(RwLock::new(vec![0u32; pixels]));

        let running = Arc::new((Mutex::new(false), Condvar::new()));
        all_running.push(running.clone());

        let scene_ref = scene.clone();
        let buf_ref = buf.clone();
        worker_handles.push(thread::spawn(move || {
            let mut rng = SmallRng::from_entropy();

            loop {
                // Wait for signal to run.
                let _ = running
                    .1
                    .wait_while(running.0.lock().unwrap(), |r| !*r)
                    .unwrap();

                let s = scene_ref.read().unwrap();
                let mut b = buf_ref.write().unwrap();
                for j in 0..pixels {
                    let pixel = i * pixels_per_worker + j;
                    let x = pixel % WIDTH;
                    let y = pixel / HEIGHT;

                    let y_unit = ((y as f64) - (HEIGHT as f64) / 2.) / (HEIGHT as f64) * -2.;
                    let x_unit = ((x as f64) - (WIDTH as f64) / 2.) / (HEIGHT as f64) * 2.;
                    let ray = s.camera.ray_for_pixel(x_unit, y_unit);

                    let c = s.ray_trace(&ray, &mut rng);
                    let c = gamma_encode(c);
                    let c = ((f32_to_u8(c.x) as u32) << 16)
                        | ((f32_to_u8(c.y) as u32) << 8)
                        | (f32_to_u8(c.z) as u32);
                    b[j] = c;
                }

                *running.0.lock().unwrap() = false;
                running.1.notify_one();
            }
        }));
        worker_bufs.push(buf);
    }

    window.limit_update_rate(Some(Duration::from_millis(16)));
    while window.is_open() {
        theta += 0.05f32;

        {
            let mut s = scene.write().unwrap();
            s.spheres[0].center.x = theta.cos() as f64;
            s.spheres[0].center.y = theta.sin() as f64;
        }

        for pair in all_running.iter() {
            // Tell workers to run.
            *pair.0.lock().unwrap() = true;
            pair.1.notify_one();
        }

        // Wait for all the workers to finish and copy the result to the window buffer.
        let mut cursor = 0;
        for i in 0..WORKER_COUNT {
            let pair = &all_running[i];
            let _ = pair.1.wait_while(pair.0.lock().unwrap(), |r| *r).unwrap();

            let b = worker_bufs[i].read().unwrap();
            buf[cursor..(cursor + b.len())].copy_from_slice(&b);
            cursor += b.len();
        }

        window
            .update_with_buffer(&buf, WIDTH, HEIGHT)
            .expect("failed to update the window's content");
    }
}
