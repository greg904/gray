mod sphere;
mod spherical;
mod triangle;
mod view_tiles;

use sphere::Sphere;
use triangle::Triangle;

use std::convert::TryFrom;
use std::convert::TryInto;
use std::f32;
use std::f64;
use std::sync::Arc;
use std::sync::Condvar;
use std::sync::Mutex;
use std::sync::RwLock;
use std::thread;
use std::time::Instant;

use clap::ArgEnum;
use clap::Parser;

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
const SAMPLES: usize = 200;
const WORKER_COUNT: usize = 8;
const TILE_SIZE: u16 = 64;
const BACKGROUND: (f32, f32, f32) = (0.471, 0.796, 0.957);

pub struct Ray {
    pub origin: DVec3,
    pub dir: DVec3,
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
    pub fn from_origin_and_random_direction_in_hemisphere(
        origin: DVec3,
        center: DVec3,
        rng: &mut SmallRng,
    ) -> Self {
        Self {
            origin,
            dir: random_unit_vector_in_hemisphere(center, rng),
        }
    }

    pub fn point_from_t(&self, t: f64) -> DVec3 {
        self.origin + t * self.dir
    }
}

struct Material {
    albedo: Vec3,
}

struct TriangleObject {
    tri: Triangle,
    mat: Material,
}

struct SphereObject {
    sph: Sphere,
    mat: Material,
}

struct Light {
    pos: DVec3,
    power: Vec3,
}

struct Camera {
    center: DVec3,
    // Precomputed values.
    grid_center: DVec3,
    grid_right: DVec3,
    grid_up: DVec3,
}

impl Camera {
    fn new(center: DVec3, dir: DVec3, z_near: f64, vertical_fov: f64) -> Self {
        let grid_center = center + z_near * dir;
        let grid_scale = 2. * (vertical_fov / 2.).tan() * z_near;
        let (grid_down, grid_left) = spherical::polar_and_azimuthal_vectors_from_radial_vector(dir);
        let grid_right = -grid_scale * grid_left;
        let grid_up = -grid_scale * grid_down;
        Self {
            center,
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
    spheres: Vec<SphereObject>,
    triangles: Vec<TriangleObject>,
    lights: Vec<Light>,
    camera: Camera,
    aim_rays: bool,
}

enum HitObjectInfo<'a> {
    Sphere(&'a SphereObject),
    Triangle {
        triangle: &'a TriangleObject,
        uv: DVec2,
        xyz: DVec3,
    },
    Nothing,
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
                let normal = (intersection - sphere.sph.center()).normalize();
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

impl Scene {
    fn first_intersection_with_ray(&self, ray: &Ray) -> Hit {
        let mut first = Hit {
            t: f64::NAN,
            obj_info: HitObjectInfo::Nothing,
        };

        for sphere in self.spheres.iter() {
            let t = match sphere.sph.t_of_intersection_point_with_ray(ray) {
                Some(val) => val,
                None => continue,
            };
            if t >= first.t {
                continue;
            }
            first = Hit {
                t,
                obj_info: HitObjectInfo::Sphere(sphere),
            };
        }

        for triangle in self.triangles.iter() {
            let t = triangle.tri.t_of_intersection_point_with_ray(ray);
            if t.is_nan() || t < 0. {
                continue;
            }
            if t >= first.t {
                continue;
            }
            if ray.dir.dot(triangle.tri.normal()) >= 0. {
                // The triangle can only be seen when its normal is pointing
                // toward us.
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
                if let Some(t) = sphere.sph.t_of_intersection_point_with_ray(&ray) {
                    if t <= light_t {
                        continue 'light_iter;
                    }
                }
            }
            for triangle in self.triangles.iter() {
                let t = triangle.tri.t_of_intersection_point_with_ray(&ray);
                if t.is_nan() || t > light_t {
                    continue;
                }
                if ray.dir.dot(triangle.tri.normal()) >= 0. {
                    // The triangle can only be seen when its normal is pointing
                    // toward us.
                    continue;
                }
                let xyz = ray.point_from_t(t);
                let uv = triangle.tri.uv_of_point(xyz);
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

    fn compute_indirect_lighting(
        &self,
        point: DVec3,
        normal: DVec3,
        rng: &mut SmallRng,
        recursion_level: u8,
    ) -> Vec3 {
        let next_ray = Ray::from_origin_and_random_direction_in_hemisphere(point, normal, rng);
        // Compute the cosine of the angle between the ray and the surface normal.
        let cos_theta = next_ray.dir.dot(normal) as f32;
        self.ray_trace_indirect(&next_ray, rng, recursion_level + 1) * cos_theta
    }

    fn ray_trace_indirect(&self, ray: &Ray, rng: &mut SmallRng, recursion_level: u8) -> Vec3 {
        if recursion_level > 2 {
            return Vec3::new(BACKGROUND.0, BACKGROUND.1, BACKGROUND.2);
        }

        let closest_hit = self.first_intersection_with_ray(ray);
        if closest_hit.t.is_nan() {
            return Vec3::new(BACKGROUND.0, BACKGROUND.1, BACKGROUND.2);
        };
        let (intersection, normal, albedo) = closest_hit.intersection_normal_albedo(ray);
        let direct_lighting = self.compute_direct_lighting(intersection, normal);
        let indirect_lighting =
            self.compute_indirect_lighting(intersection, normal, rng, recursion_level);

        albedo * (direct_lighting + indirect_lighting)
    }

    fn ray_trace(&self, ray: &Ray, rng: &mut SmallRng) -> Vec3 {
        let closest_hit = self.first_intersection_with_ray(ray);
        if closest_hit.t.is_nan() {
            return Vec3::new(BACKGROUND.0, BACKGROUND.1, BACKGROUND.2);
        };
        let (intersection, normal, albedo) = closest_hit.intersection_normal_albedo(ray);
        let direct_lighting = self.compute_direct_lighting(intersection, normal);

        let mut indirect_lighting = Vec3::ZERO;

        if self.aim_rays {
            let mut area_by_index: [(usize, f32); SAMPLES] = [(usize::MAX, 0.); SAMPLES];
            let mut cursor = 0;
            let mut total_area = 0.;

            let mut indirect_lighting_area = 0.;

            for (i, sphere) in self.spheres.iter().enumerate() {
                // The sphere is visible from our side.
                if normal.dot(sphere.sph.center() - intersection) <= 0. {
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

            for (i, triangle) in self.triangles.iter().enumerate() {
                // The triangle is visible from our side.
                if triangle.tri.normal().dot(triangle.tri.a() - intersection) >= 0. {
                    continue;
                }
                let a_ = (triangle.tri.a() - intersection).normalize();
                let b_ = (triangle.tri.b() - intersection).normalize();
                let c_ = (triangle.tri.c() - intersection).normalize();
                let area = spherical::area_of_intersection_of_spherical_triangle_and_unit_hemisphere(
                    a_, b_, c_, 1., normal,
                ) as f32;
                if area <= 0.00001 {
                    continue;
                }
                area_by_index[cursor] = (self.spheres.len() + i, area);
                cursor += 1;
                total_area += area;
            }

            if cursor > 0 {
                for i in 0..SAMPLES {
                    let j = i % cursor;
                    let divisor = ((SAMPLES - j) / cursor) as f32;
                    let obj = area_by_index[j].0;
                    let next_ray_dir = if obj < self.spheres.len() {
                        let sphere = &self.spheres[obj];
                        let target = spherical::random_look_in_sphere(
                            intersection,
                            sphere.sph.center(),
                            sphere.sph.radius(),
                            rng,
                        );
                        (target - intersection).normalize()
                    } else {
                        let triangle = &self.triangles[obj - self.spheres.len()];
                        spherical::random_direction_toward_triangle(
                            triangle.tri.a() - intersection,
                            triangle.tri.b() - intersection,
                            triangle.tri.c() - intersection,
                            rng,
                        )
                    };
                    let next_ray = Ray {
                        origin: intersection,
                        dir: next_ray_dir,
                    };
                    // Compute the cosine of the angle between the ray and the surface normal.
                    let cos_theta = next_ray.dir.dot(normal) as f32;
                    let intensity = self.ray_trace_indirect(&next_ray, rng, 1) * cos_theta;
                    indirect_lighting += intensity * area_by_index[j].1 / divisor;
                    indirect_lighting_area += area_by_index[j].1 / divisor;
                }
            }

            let ambient_area = 2. * f32::consts::PI - total_area;
            if ambient_area > 0. {
                let indirect_lighting_cos_theta_factor = 0.5;
                indirect_lighting += Vec3::new(BACKGROUND.0, BACKGROUND.1, BACKGROUND.2)
                    * indirect_lighting_cos_theta_factor
                    * ambient_area;
                indirect_lighting_area += ambient_area;
            }

            indirect_lighting /= indirect_lighting_area;
        } else {
            for _ in 0..SAMPLES {
                indirect_lighting += self.compute_indirect_lighting(intersection, normal, rng, 1);
            }
            indirect_lighting /= SAMPLES as f32;
        }

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

/// A hack to share a pointer between multiple threads. This is used in unsafe
/// code.
#[derive(Clone, Copy)]
struct SendMutPtr<T>(*mut T);

unsafe impl<T> Send for SendMutPtr<T> {}

#[derive(ArgEnum, Copy, Clone)]
enum ToneMap {
    GammaEncode,
    AcesLike,
}

#[derive(Parser)]
struct Args {
    /// Aim rays to the primitives instead of picking random directions.
    #[clap(long)]
    aim_rays: bool,

    /// The tone mapping mode.
    #[clap(long, arg_enum, default_value = "gamma-encode")]
    tone_map: ToneMap,
}

fn main() {
    let args = Args::parse();
    let tone_map = args.tone_map;

    let mut window = Window::new("gray", WIDTH, HEIGHT, WindowOptions::default())
        .expect("failed to create window");

    let scene = Arc::new(RwLock::new(Scene {
        spheres: vec![SphereObject {
            sph: Sphere::new(DVec3::ZERO, 0.7),
            mat: Material {
                albedo: Vec3::new(1., 0., 0.),
            },
        }],
        triangles: vec![
            TriangleObject {
                tri: Triangle::new((
                    DVec3::new(-50., -50., -0.7),
                    DVec3::new(50., -50., -0.7),
                    DVec3::new(-50., 50., -0.7),
                )),
                mat: Material {
                    albedo: Vec3::new(1., 1., 1.),
                },
            },
            TriangleObject {
                tri: Triangle::new((
                    DVec3::new(50., 50., -0.7),
                    DVec3::new(-50., 50., -0.7),
                    DVec3::new(50., -50., -0.7),
                )),
                mat: Material {
                    albedo: Vec3::new(1., 1., 1.),
                },
            },
        ],
        lights: vec![Light {
            pos: DVec3::new(10., 17., 50.),
            power: Vec3::new(1000., 1000., 1000.),
        }],
        camera: Camera::new(
            DVec3::new(-1.2, 0., 0.3),
            DVec3::new(1., 0., -0.2).normalize(),
            0.1,
            80f64.to_radians(),
        ),
        aim_rays: args.aim_rays,
    }));

    let tiles = Arc::new(view_tiles::Tiles::new(
        WIDTH.try_into().unwrap(),
        HEIGHT.try_into().unwrap(),
        TILE_SIZE,
    ));

    // These are the tiles not rendered yet.
    let tile_queue: Arc<Mutex<Vec<u16>>> =
        Arc::new(Mutex::new(Vec::with_capacity(tiles.tile_count().into())));
    let tile_enqueued = Arc::new(Condvar::new());
    let tile_dequeued = Arc::new(Condvar::new());

    let mut pixels: Vec<u32> = vec![0; WIDTH * HEIGHT];
    let pixels_ptr = SendMutPtr(pixels.as_mut_ptr());

    let mut worker_handles = Vec::new();
    let mut workers_running = Vec::new();
    for _ in 0..WORKER_COUNT {
        let running = Arc::new((Mutex::new(false), Condvar::new()));
        workers_running.push(running.clone());

        let tiles_ref = tiles.clone();
        let tile_queue_ref = tile_queue.clone();
        let tile_enqueued_ref = tile_enqueued.clone();
        let tile_dequeued_ref = tile_dequeued.clone();
        let scene_ref = scene.clone();

        worker_handles.push(thread::spawn(move || {
            let mut rng = SmallRng::from_entropy();

            loop {
                // Wait for a tile and remove it from the queue.
                let tile_index = {
                    let mut q = tile_queue_ref.lock().unwrap();
                    let i = loop {
                        q = match q.pop() {
                            Some(val) => break val,
                            None => tile_enqueued_ref.wait(q).unwrap(),
                        };
                    };
                    *running.0.lock().unwrap() = true;
                    i
                };
                let tile = tiles_ref.get(tile_index);

                tile_dequeued_ref.notify_one();

                let s = scene_ref.read().unwrap();

                for y in tile.y_min..tile.y_max {
                    for x in tile.x_min..tile.x_max {
                        let x_unit = ((x as f64) - (WIDTH as f64) / 2.) / (HEIGHT as f64) * 2.;
                        let y_unit = ((y as f64) - (HEIGHT as f64) / 2.) / (HEIGHT as f64) * -2.;
                        let ray = s.camera.ray_for_pixel(x_unit, y_unit);

                        let c = s.ray_trace(&ray, &mut rng);
                        let c = match tone_map {
                            ToneMap::GammaEncode => gamma_encode(c),
                            ToneMap::AcesLike => aces_fit(c),
                        };
                        let c = ((f32_to_u8(c.x) as u32) << 16)
                            | ((f32_to_u8(c.y) as u32) << 8)
                            | (f32_to_u8(c.z) as u32);

                        let i = isize::try_from(y).unwrap() * isize::try_from(WIDTH).unwrap()
                            + isize::try_from(x).unwrap();
                        unsafe { *pixels_ptr.0.offset(i) = c };
                    }
                }

                *running.0.lock().unwrap() = false;
                running.1.notify_one();
            }
        }));
    }

    let mut theta = 0f64;

    let mut timer = Instant::now();
    let mut render_cnt = 0;

    while window.is_open() {
        if render_cnt >= 40 {
            let now = Instant::now();
            eprintln!(
                "render time: {}ms",
                ((now - timer) / render_cnt).as_millis()
            );
            timer = now;
            render_cnt = 0;
        }
        render_cnt += 1;

        theta += 0.05;

        {
            let mut s = scene.write().unwrap();
            s.spheres[0].sph.center_mut().y = 0.3 * theta.cos();
        }

        {
            let mut t = tile_queue.lock().unwrap();
            for i in 0..tiles.tile_count() {
                t.push(i);
            }
            tile_enqueued.notify_all();
        }

        // Wait until no tile is left and all the workers are done.
        let _ = tile_dequeued
            .wait_while(tile_queue.lock().unwrap(), |q| !q.is_empty())
            .unwrap();
        for i in 0..WORKER_COUNT {
            let pair = &workers_running[i];
            let _ = pair.1.wait_while(pair.0.lock().unwrap(), |r| *r).unwrap();
        }

        window
            .update_with_buffer(&pixels, WIDTH, HEIGHT)
            .expect("failed to update the window's content");
    }
}
