//! Rasterization helpers.

use core::convert::TryInto;
use core::mem;
use core::ops::Range;

use glam::IVec2;
use glam::Vec3;
use glam::Vec3Swizzles;

pub struct Rasterizer {
    view_width: i32,
    view_height: i32,
    color: Vec<u32>,
    z: Vec<f32>,

    z_near: f32,
    z_far: f32,
    vertical_fov: f32,
}

fn range_intersection(a: Range<i32>, b: Range<i32>) -> Range<i32> {
    if a.end <= b.start || b.end <= a.start {
        // The intersection is empty.
        Range { start: 0, end: 0 }
    } else {
        Range {
            start: a.start.max(b.start),
            end: a.end.min(b.end),
        }
    }
}

impl Rasterizer {
    pub fn new(view_width: usize, view_height: usize) -> Rasterizer {
        let pixel_cnt = view_width * view_height;
        Rasterizer {
            view_width: view_width.try_into().unwrap(),
            view_height: view_height.try_into().unwrap(),
            color: vec![0; pixel_cnt],
            z: vec![1.; pixel_cnt],

            z_near: 0.1,
            z_far: 10000.,
            vertical_fov: 80.,
        }
    }

    fn camera_space_to_view_space(&self, camera_space: Vec3) -> IVec2 {
        let mut view_space = camera_space.xy();
        // Project onto the view, still in camera space.
        view_space *= self.z_near / camera_space.z;
        // Convert into view space.
        view_space /= self.vertical_fov.tan() * self.z_near;
        view_space *= (self.view_height as f32) / 2.;
        view_space.x += (self.view_width as f32) / 2.;
        view_space.y += (self.view_height as f32) / 2.;
        view_space.as_ivec2()
    }

    fn camera_space_to_view_space_depth(&self, z: f32) -> f32 {
        (z - self.z_near) / (self.z_far - self.z_near)
    }

    fn fill_triangle_custom<F>(&mut self, mut p0: Vec3, mut p1: Vec3, mut p2: Vec3, mut f: F)
    where
        F: FnMut(&mut Self, i32, i32, Vec3),
    {
        let mut p0v = self.camera_space_to_view_space(p0);
        let mut p1v = self.camera_space_to_view_space(p1);
        let mut p2v = self.camera_space_to_view_space(p2);

        // Sort the vertices by their y coordinate in the view space, because we will always render
        // from top to bottom.
        if p1v.y < p0v.y {
            mem::swap(&mut p0, &mut p1);
            mem::swap(&mut p0v, &mut p1v);
        }
        if p2v.y < p1v.y {
            mem::swap(&mut p1, &mut p2);
            mem::swap(&mut p1v, &mut p2v);
        }
        if p1v.y < p0v.y {
            mem::swap(&mut p0, &mut p1);
            mem::swap(&mut p0v, &mut p1v);
        }

        let mut fill_horizontal_segment = |this: &mut Self, y, mut x0, mut x1| {
            if x1 < x0 {
                mem::swap(&mut x0, &mut x1);
            }

            for x in range_intersection(x0..(x1 + 1), 0..this.view_width) {
                let p = IVec2::new(x, y);
                let b_denom = (p1v - p0v).dot(p2v - p1v) as f32 / 2.;
                let b0 = (p1v - p).dot(p2v - p) as f32 / 2. / b_denom;
                let b1 = (p - p0v).dot(p2v - p0v) as f32 / 2. / b_denom;
                let b2 = (p1v - p0v).dot(p - p0v) as f32 / 2. / b_denom;
                // Get the perspective correct barycentric.
                let pc = Vec3::new(b0 / p0.z, b1 / p1.z, b2 / p2.z)
                    / (b0 / p0.z + b1 / p1.z + b2 / p2.z);
                f(this, x, y, pc);
            }
        };

        for y in range_intersection(p0v.y..p1v.y, 0..self.view_height) {
            let x0 = p0v.x + (p1v.x - p0v.x) * (y - p0v.y) / (p1v.y - p0v.y);
            let x1 = p0v.x + (p2v.x - p0v.x) * (y - p0v.y) / (p2v.y - p0v.y);
            fill_horizontal_segment(self, y, x0, x1);
        }

        for y in range_intersection(p1v.y..(p2v.y + 1), 0..self.view_height) {
            let x0 = p1v.x + (p2v.x - p1v.x) * (y - p1v.y) / (p1v.y - p0v.y);
            let x1 = p0v.x + (p2v.x - p0v.x) * (y - p0v.y) / (p2v.y - p0v.y);
            fill_horizontal_segment(self, y, x0, x1);
        }
    }

    pub fn rasterize_triangle(&mut self, p0: Vec3, p1: Vec3, p2: Vec3, color: u32) {
        self.fill_triangle_custom(p0, p1, p2, |this, x, y, b| {
            let i = (y * this.view_width + x) as usize;
            let z = Vec3::new(p0.z, p1.z, p2.z).dot(b);
            if z < this.z_near || z > this.z_far {
                return;
            }
            let z = this.camera_space_to_view_space_depth(z);
            // Z test.
            if z < this.z[i] {
                this.color[i] = color;
                this.z[i] = z;
            }
        });
    }

    pub fn clear(&mut self) {
        self.color.fill(0);
        self.z.fill(1.);
    }

    pub fn color(&self) -> &[u32] {
        &self.color
    }

    pub fn z(&self) -> &[f32] {
        &self.z
    }
}
