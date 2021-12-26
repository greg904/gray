//! Rasterization helpers.

use core::convert::TryInto;
use core::mem;
use core::ops::Range;

use glam::IVec2;
use glam::Vec3A;

pub struct Rasterizer {
    view_width: i32,
    view_height: i32,
    color: Vec<u32>,
    z: Vec<f32>,

    z_near: f32,
    view_space_factor: f32,
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

fn clip_triangle<F>(mut p0: Vec3A, mut p1: Vec3A, mut p2: Vec3A, z_near: f32, mut f: F)
where
    F: FnMut(Vec3A, Vec3A, Vec3A),
{
    // Sort the vertices by their Z coordinate.
    if p1.z < p0.z {
        mem::swap(&mut p0, &mut p1);
    }
    if p2.z < p1.z {
        mem::swap(&mut p1, &mut p2);
    }
    if p1.z < p0.z {
        mem::swap(&mut p0, &mut p1);
    }

    // The whole triangle is behind the Z near plane.
    if p0.z > -z_near {
        return;
    }

    if p2.z > -z_near {
        // A part of the triangle is behind the Z near plane.
        let p2_to_p0 = p0.lerp(p2, (-z_near - p0.z) / (p2.z - p0.z));
        if p1.z > -z_near {
            let p1_to_p0 = p0.lerp(p1, (-z_near - p0.z) / (p1.z - p0.z));
            f(p0, p1_to_p0, p2_to_p0);
        } else {
            let p2_to_p1 = p1.lerp(p2, (-z_near - p1.z) / (p2.z - p1.z));
            f(p0, p1, p2_to_p0);
            f(p1, p2_to_p0, p2_to_p1);
        }
    } else {
        // The whole triangle is in front of the Z near plane.
        f(p0, p1, p2);
    }
}

impl Rasterizer {
    pub fn new(
        view_width: usize,
        view_height: usize,
        z_near: f32,
        vertical_fov: f32,
    ) -> Rasterizer {
        let pixel_cnt = view_width * view_height;
        let camera_height = 2. * (vertical_fov / 2.).tan();
        Rasterizer {
            view_width: view_width.try_into().unwrap(),
            view_height: view_height.try_into().unwrap(),
            color: vec![0; pixel_cnt],
            z: vec![f32::NEG_INFINITY; pixel_cnt],

            z_near,
            view_space_factor: (view_height as f32) / camera_height,
            vertical_fov,
        }
    }

    fn camera_space_to_view_space(&self, camera_space: Vec3A) -> IVec2 {
        let mut view_space = camera_space.truncate();
        view_space *= -self.view_space_factor / camera_space.z;
        view_space.y = -view_space.y;
        view_space.x += (self.view_width as f32) / 2.;
        view_space.y += (self.view_height as f32) / 2.;
        view_space.as_ivec2()
    }

    fn fill_triangle_custom<F>(
        &mut self,
        unclipped_p0: Vec3A,
        unclipped_p1: Vec3A,
        unclipped_p2: Vec3A,
        mut f: F,
    ) where
        F: FnMut(&mut Self, i32, i32, f32),
    {
        clip_triangle(
            unclipped_p0,
            unclipped_p1,
            unclipped_p2,
            self.z_near,
            |mut p0, mut p1, mut p2| {
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

                let mut fill_horizontal_segment = |this: &mut Self, y, x0, x1| {
                    for x in range_intersection(x0..(x1 + 1), 0..this.view_width) {
                        let p = IVec2::new(x, y);
                        // From https://gamedev.stackexchange.com/a/23745.
                        let v0 = (p1v - p0v).as_vec2();
                        let v1 = (p2v - p0v).as_vec2();
                        let v2 = (p - p0v).as_vec2();
                        let d00 = v0.dot(v0);
                        let d01 = v0.dot(v1);
                        let d11 = v1.dot(v1);
                        let d20 = v2.dot(v0);
                        let d21 = v2.dot(v1);
                        let denom = d00 * d11 - d01 * d01;
                        let b1 = (d11 * d20 - d01 * d21) / denom;
                        let b2 = (d00 * d21 - d01 * d20) / denom;
                        let b0 = 1. - b1 - b2;
                        // Get the perspective correct barycentric.
                        let pc = Vec3A::new(b0 / p0.z, b1 / p1.z, b2 / p2.z)
                            / (b0 / p0.z + b1 / p1.z + b2 / p2.z);
                        f(this, x, y, pc.x * p0.z + pc.y * p1.z + pc.z * p2.z);
                    }
                };

                if p1v.x <= p2v.x {
                    for y in range_intersection(p0v.y..p1v.y, 0..self.view_height) {
                        let x0 = p0v.x + (p1v.x - p0v.x) * (y - p0v.y) / (p1v.y - p0v.y);
                        let x1 = p0v.x + (p2v.x - p0v.x) * (y - p0v.y) / (p2v.y - p0v.y);
                        fill_horizontal_segment(self, y, x0, x1);
                    }
                } else {
                    for y in range_intersection(p0v.y..p1v.y, 0..self.view_height) {
                        let x0 = p0v.x + (p1v.x - p0v.x) * (y - p0v.y) / (p1v.y - p0v.y);
                        let x1 = p0v.x + (p2v.x - p0v.x) * (y - p0v.y) / (p2v.y - p0v.y);
                        fill_horizontal_segment(self, y, x1, x0);
                    }
                }

                if p1v.x <= p0v.x + (p2v.x - p0v.x) * (p1v.y - p0v.y) {
                    for y in range_intersection(p1v.y..p2v.y, 0..self.view_height) {
                        let x0 = p1v.x + (p2v.x - p1v.x) * (y - p1v.y) / (p2v.y - p1v.y);
                        let x1 = p0v.x + (p2v.x - p0v.x) * (y - p0v.y) / (p2v.y - p0v.y);
                        fill_horizontal_segment(self, y, x0, x1);
                    }
                } else {
                    for y in range_intersection(p1v.y..p2v.y, 0..self.view_height) {
                        let x0 = p1v.x + (p2v.x - p1v.x) * (y - p1v.y) / (p2v.y - p1v.y);
                        let x1 = p0v.x + (p2v.x - p0v.x) * (y - p0v.y) / (p2v.y - p0v.y);
                        fill_horizontal_segment(self, y, x1, x0);
                    }
                }

                // Initially, I had the second range above go to `p2v.y + 1` because the last pixel is
                // included. However, for a triangle where `p1v.y == p2v.y`, this would result in a
                // division by zero. The fix is to let the range be exclusive so that it won't run in that
                // case and do the last point manually here.
                if p2v.x >= 0 && p2v.x < self.view_width && p2v.y >= 0 && p2v.y < self.view_height {
                    f(self, p2v.x, p2v.y, p2.z);
                }
            },
        )
    }

    pub fn rasterize_triangle(&mut self, p0: Vec3A, p1: Vec3A, p2: Vec3A, color: u32) {
        self.fill_triangle_custom(p0, p1, p2, |this, x, y, z| {
            let i = (y * this.view_width + x) as usize;
            if z > this.z[i] {
                this.color[i] = color;
                this.z[i] = z;
            }
        });
    }

    pub fn rasterize_sphere(&mut self, center: Vec3A, r: f32, color: u32) {
        // A point on the projection of the sphere to the Z near plane satisfies the following equation:
        // (r^2 - y_O^2 - z_O^2)x^2 + 2x_Oy_Oxy + (r^2 - x_O^2 - z_O^2)y^2 - 2z_Oz_Nx_Ox - 2z_Oz_Ny_Oy - z_N^2(x_O^2 + y_O^2 - r^2) >= 0
        // TODO: find a way to draw this conic more efficiently
        let grid_size = self.z_near * (self.vertical_fov / 2.).tan();
        let center_sqr = center * center;
        let r_sqr = r * r;
        let r_sqr_minus_center_sqr_y_minus_center_sqr_z = r_sqr - center_sqr.y - center_sqr.z;
        let two_center_x_center_y = 2. * center.x * center.y;
        let r_sqr_minus_center_sqr_x_minus_center_sqr_z = r_sqr - center_sqr.x - center_sqr.z;
        let two_center_z_z_near = 2. * center.z * self.z_near;
        let two_center_z_z_near_center_x = two_center_z_z_near * center.x;
        let two_center_z_z_near_center_y = two_center_z_z_near * center.y;
        let z_near_sqr_times_center_sqr_x_plus_center_sqr_y_minus_r_sqr =
            self.z_near * self.z_near * (center_sqr.x + center_sqr.y - r_sqr);
        for y in 0..self.view_height {
            for x in 0..self.view_width {
                let x_unit =
                    2. * (x as f32 - self.view_width as f32 / 2.) / self.view_height as f32;
                let y_unit =
                    -2. * (y as f32 - self.view_height as f32 / 2.) / self.view_height as f32;
                let x_near = x_unit * grid_size;
                let y_near = y_unit * grid_size;

                let quarter_discriminant =
                    r_sqr_minus_center_sqr_y_minus_center_sqr_z * x_near * x_near
                        + two_center_x_center_y * x_near * y_near
                        + r_sqr_minus_center_sqr_x_minus_center_sqr_z * y_near * y_near
                        - two_center_z_z_near_center_x * x_near
                        - two_center_z_z_near_center_y * y_near
                        - z_near_sqr_times_center_sqr_x_plus_center_sqr_y_minus_r_sqr;
                if quarter_discriminant < 0. {
                    continue;
                }

                let a = x_near * x_near + y_near * y_near + self.z_near * self.z_near;
                let b = -2. * (x_near * center.x + y_near * center.y - self.z_near * center.z);
                let _c = center.x * center.x + center.y * center.y + center.z * center.z - r * r;
                let quarter_discriminant_sqrt = quarter_discriminant.sqrt();
                let t = (-b - 2. * quarter_discriminant_sqrt) / (2. * a);
                let z = -self.z_near * t;

                let i = (y * self.view_width + x) as usize;
                if z > self.z[i] {
                    self.color[i] = color;
                    self.z[i] = z;
                }
            }
        }
    }

    pub fn clear(&mut self, clear_color: u32) {
        self.color.fill(clear_color);
        self.z.fill(f32::NEG_INFINITY);
    }

    pub fn color(&self) -> &[u32] {
        &self.color
    }

    pub fn z(&self) -> &[f32] {
        &self.z
    }
}
