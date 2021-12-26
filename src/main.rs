mod light;
mod rasterize;
mod ray_trace;
mod scene;
mod sphere;
mod spherical;
mod triangle;
mod view_tiles;

use light::Light;
use rasterize::Rasterizer;
use ray_trace::RayTracer;
use scene::Material;
use scene::Scene;
use scene::SphereObject;
use scene::TriangleObject;
use sphere::Sphere;
use triangle::Triangle;

use std::convert::TryFrom;
use std::convert::TryInto;
use std::f32;
use std::sync::Arc;
use std::sync::Condvar;
use std::sync::Mutex;
use std::sync::RwLock;
use std::thread;
use std::time::Instant;

use clap::ArgEnum;
use clap::Parser;

use glam::Affine3A;
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
const WORKER_COUNT: usize = 8;
const TILE_SIZE: u16 = 64;
const BACKGROUND: (f32, f32, f32) = (0.471, 0.796, 0.957);

pub struct Ray {
    pub origin: Vec3,
    pub dir: Vec3,
}

pub fn random_unit_vector_in_hemisphere_and_cos_theta(center: Vec3, rng: &mut SmallRng) -> (Vec3, f32) {
    let v = UnitSphere.sample(rng);
    let v = Vec3::new(v[0], v[1], v[2]);
    let dot = v.dot(center);
    let sign = 1f32.copysign(dot);
    // Make sure that the vector is in the hemisphere.
    (v / sign, dot / sign)
}

impl Ray {
    pub fn point_from_t(&self, t: f32) -> Vec3 {
        self.origin + t * self.dir
    }
}

pub struct Camera {
    center: Vec3,
    // Precomputed values for ray-tracing.
    grid_center: Vec3,
    grid_right: Vec3,
    grid_up: Vec3,
    // Precomputed value for rasterization.
    transform: Affine3A,
}

impl Camera {
    fn new(center: Vec3, dir: Vec3, z_near: f32, vertical_fov: f32) -> Self {
        let grid_center = center + z_near * dir;
        let grid_scale = (vertical_fov / 2.).tan() * z_near;
        let (d, l) = spherical::polar_and_azimuthal_vectors_from_radial_vector(dir);
        let r = -l;
        let u = -d;
        let grid_right = grid_scale * r;
        let grid_up = grid_scale * u;
        let rot_matrix = Mat3::from_cols(r, u, -dir).transpose();
        Self {
            center,
            grid_center,
            grid_right,
            grid_up,
            transform: Affine3A::from_mat3(rot_matrix) * Affine3A::from_translation(-center),
        }
    }

    /// Returns the ray that needs to be shot to give the color for a pixel.
    ///
    /// `y` must be between `-1` and `1` inclusive, while `x` is in the
    /// corresponding range such that the aspect ratio is respected. For
    /// example, with an aspect ratio of `2:1`, then `x` lies between
    /// `-2` and `2` inclusive.
    fn ray_for_pixel(&self, x: f32, y: f32) -> Ray {
        let pixel = self.grid_center + x * self.grid_right + y * self.grid_up;
        Ray {
            origin: pixel,
            dir: (pixel - self.center).normalize(),
        }
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
    Linear,
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

    /// The amount of samples per pixel.
    #[clap(short, long, default_value = "5")]
    samples: usize,

    /// The amount of light bounces per sample.
    #[clap(short, long, default_value = "3")]
    bounces: u8,

    /// Use rasterization instead of the first ray.
    #[clap(short, long)]
    rasterize: bool,

    /// Pre-compute the average diffuse of primitives and use it for approximations.
    #[clap(long)]
    avg_diffuse: bool,
}

fn main() {
    let args = Args::parse();
    let tone_map = args.tone_map;
    let rasterize = args.rasterize;
    let aim_rays = args.aim_rays;
    let bounces = args.bounces;
    let samples = args.samples;
    let avg_diffuse = args.avg_diffuse;

    let mut window = Window::new("gray", WIDTH, HEIGHT, WindowOptions::default())
        .expect("failed to create window");

    let scene = Arc::new(RwLock::new(Scene {
        spheres: vec![SphereObject {
            sph: Sphere::new(Vec3::ZERO, 0.7),
            mat: Material {
                albedo: Vec3::new(1., 0., 0.),
            },
        }],
        triangles: vec![
            TriangleObject {
                tri: Triangle::new((
                    Vec3::new(-50., -50., -0.7),
                    Vec3::new(50., -50., -0.7),
                    Vec3::new(-50., 50., -0.7),
                )),
                mat: Material {
                    albedo: Vec3::new(1., 1., 1.),
                },
            },
            TriangleObject {
                tri: Triangle::new((
                    Vec3::new(50., 50., -0.7),
                    Vec3::new(-50., 50., -0.7),
                    Vec3::new(50., -50., -0.7),
                )),
                mat: Material {
                    albedo: Vec3::new(1., 1., 1.),
                },
            },
        ],
        primitives_avg_diffuse: vec![Vec3::ZERO; 3],
        lights: vec![Light::new(
            Vec3::new(10., 17., 50.),
            Vec3::new(8000., 8000., 8000.),
        )],
        camera: Camera::new(
            Vec3::new(-1.2, 0., 0.3),
            Vec3::new(1., 0., -0.2).normalize(),
            0.1,
            80f32.to_radians(),
        ),
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

    let rasterizer = Arc::new(RwLock::new(Rasterizer::new(
        WIDTH,
        HEIGHT,
        0.1,
        80f32.to_radians(),
    )));

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
        let rasterizer_ref = rasterizer.clone();

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
                let r = rasterizer_ref.read().unwrap();

                for y in tile.y_min..tile.y_max {
                    for x in tile.x_min..tile.x_max {
                        let i = isize::try_from(y).unwrap() * isize::try_from(WIDTH).unwrap()
                            + isize::try_from(x).unwrap();

                        let c = if rasterize {
                            let obj = r.color()[i as usize] as usize;
                            if obj < s.triangles.len() {
                                s.triangles[obj].mat.albedo
                            } else if obj < s.triangles.len() + s.spheres.len() {
                                s.spheres[obj - s.triangles.len()].mat.albedo
                            } else {
                                Vec3::new(BACKGROUND.0, BACKGROUND.1, BACKGROUND.2)
                            }
                        } else {
                            let x_unit = ((x as f32) - (WIDTH as f32) / 2.) / (HEIGHT as f32) * 2.;
                            let y_unit =
                                ((y as f32) - (HEIGHT as f32) / 2.) / (HEIGHT as f32) * -2.;
                            let ray = s.camera.ray_for_pixel(x_unit, y_unit);
                            let ray_tracer = RayTracer {
                                aim_rays,
                                bounces,
                                samples,
                                use_avg_diffuse: avg_diffuse,
                                scene: &s,
                            };
                            ray_tracer.ray_trace(&ray, &mut rng)
                        };

                        let c = match tone_map {
                            ToneMap::Linear => c,
                            ToneMap::GammaEncode => gamma_encode(c),
                            ToneMap::AcesLike => aces_fit(c),
                        };
                        let c = ((f32_to_u8(c.x) as u32) << 16)
                            | ((f32_to_u8(c.y) as u32) << 8)
                            | (f32_to_u8(c.z) as u32);

                        unsafe { *pixels_ptr.0.offset(i) = c };
                    }
                }

                *running.0.lock().unwrap() = false;
                running.1.notify_one();
            }
        }));
    }

    let mut theta = 0f32;

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

            if args.avg_diffuse {
                s.compute_primitives_avg_diffuse();
            }
        }

        if rasterize {
            let mut r = rasterizer.write().unwrap();
            r.clear(u32::MAX);

            let s = scene.read().unwrap();

            for (i, triangle) in s.triangles.iter().enumerate() {
                r.rasterize_triangle(
                    s.camera.transform.transform_point3(triangle.tri.a()),
                    s.camera.transform.transform_point3(triangle.tri.b()),
                    s.camera.transform.transform_point3(triangle.tri.c()),
                    i as u32,
                );
            }

            for (i, sphere) in s.spheres.iter().enumerate() {
                r.rasterize_sphere(
                    s.camera.transform.transform_point3(*sphere.sph.center()),
                    sphere.sph.radius(),
                    (i + s.triangles.len()) as u32,
                );
            }
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
