use core::f32;
use core::mem;

use glam::Vec3;

use rand::distributions::Distribution;
use rand::rngs::SmallRng;
use rand::Rng;
use rand_distr::UnitDisc;

pub fn polar_and_azimuthal_vectors_from_radial_vector(radial_vector: Vec3) -> (Vec3, Vec3) {
    let azimuthal_vector = Vec3::Z.cross(radial_vector).normalize();
    let polar_vector = azimuthal_vector.cross(radial_vector);
    (polar_vector, azimuthal_vector)
}

pub fn area_of_sphere_projected_to_unit_sphere(sphere_d: f32, sphere_r: f32) -> f32 {
    // We consider the cone tangent to the sphere.
    let r = (sphere_r / sphere_d).asin();
    2. * f32::consts::PI * (1. - r.cos())
}

pub fn area_of_intersection_of_spherical_triangle_and_unit_hemisphere(
    mut a_: Vec3,
    mut b_: Vec3,
    mut c_: Vec3,
    hemisphere_r: f32,
    hemisphere_dir: Vec3,
) -> f32 {
    let mut a_dot_hd = a_.dot(hemisphere_dir);
    let mut b_dot_hd = b_.dot(hemisphere_dir);
    let mut c_dot_hd = c_.dot(hemisphere_dir);
    if b_dot_hd < a_dot_hd {
        mem::swap(&mut a_dot_hd, &mut b_dot_hd);
        mem::swap(&mut a_, &mut b_);
    }
    if c_dot_hd < b_dot_hd {
        mem::swap(&mut b_dot_hd, &mut c_dot_hd);
        mem::swap(&mut b_, &mut c_);
    }
    if b_dot_hd < a_dot_hd {
        mem::swap(&mut a_dot_hd, &mut b_dot_hd);
        mem::swap(&mut a_, &mut b_);
    }
    if c_dot_hd <= 0. {
        return 0.;
    }
    let mut to_remove = 0.;
    if a_dot_hd < 0. && b_dot_hd < 0. {
        let d = c_ - a_;
        let good_corr = (-a_dot_hd / d.dot(hemisphere_dir)) * d;
        a_ += good_corr;
        a_ = a_.normalize();

        let d = c_ - b_;
        let good_corr = (-b_dot_hd / d.dot(hemisphere_dir)) * d;
        b_ += good_corr;
        b_ = b_.normalize();
    } else if a_dot_hd < 0. || b_dot_hd < 0. {
        // Compute the area of the spherical triangle and remove the bits not in the hemisphere.
        to_remove = area_of_intersection_of_spherical_triangle_and_unit_hemisphere(
            a_,
            b_,
            c_,
            hemisphere_r,
            -hemisphere_dir,
        );
    }
    // From https://math.stackexchange.com/a/66731.
    let a = b_.dot(c_).clamp(-1., 1.).acos();
    let b = c_.dot(a_).clamp(-1., 1.).acos();
    let c = a_.dot(b_).clamp(-1., 1.).acos();
    let s = (a + b + c) / 2.;
    let tan_e_over_4 =
        ((s / 2.).tan() * ((s - a) / 2.).tan() * ((s - b) / 2.).tan() * ((s - c) / 2.).tan())
            .sqrt();
    let e = 4. * tan_e_over_4.atan();
    let area = e * hemisphere_r * hemisphere_r;
    area - to_remove
}

pub fn random_look_in_sphere(
    from: Vec3,
    sphere_center: Vec3,
    sphere_r: f32,
    rng: &mut SmallRng,
) -> Vec3 {
    // Make sure that we are outside the sphere as expected.
    assert!(from.distance_squared(sphere_center) > sphere_r * sphere_r);
    // We sample a coordinate in a unit disc and project it onto the
    // sphere with the Pythagorean theorem.
    // TODO: Does this actually preserve the same distribution as if we were picking a ray at
    //       random and checking if it was intersecting with the sphere?
    let unit_xy: [f32; 2] = UnitDisc.sample(rng);
    let unit_z = (1. - unit_xy[0] * unit_xy[0] - unit_xy[1] * unit_xy[1]).sqrt();

    let z = (from - sphere_center).normalize();
    let (minus_y, x) = polar_and_azimuthal_vectors_from_radial_vector(z);
    let y = -minus_y;

    let sphere_point = sphere_center + sphere_r * (unit_xy[0] * x + unit_xy[1] * y + unit_z * z);
    sphere_point
}

pub fn random_direction_toward_triangle(a: Vec3, b: Vec3, c: Vec3, rng: &mut SmallRng) -> Vec3 {
    // We project the triangle's points to points on a unit sphere, and then pick a point
    // inside of the spherical triangle described by those points. As an approximation, we
    // can pick a point in the non-spherical triangle and normalize the vector instead.
    let a_ = a.normalize();
    let b_ = b.normalize();
    let c_ = c.normalize();
    // From https://blogs.sas.com/content/iml/2020/10/19/random-points-in-triangle.html.
    let mut p: f32 = rng.gen();
    let mut q: f32 = rng.gen();
    if p + q > 1. {
        p = 1. - p;
        q = 1. - q;
    }
    let m = a_ + p * (b_ - a_) + q * (c_ - a_);
    m.normalize()
}

mod tests {
    use crate::sphere::Sphere;
    use crate::triangle::Triangle;
    use crate::Ray;

    use glam::Vec3;

    use rand::rngs::SmallRng;
    use rand::Rng;
    use rand::SeedableRng;

    #[test]
    fn random_look_in_sphere() {
        let mut rng = SmallRng::seed_from_u64(0);
        for _ in 0..10 {
            let sphere_center = rng.gen::<Vec3>() * 100. - 50.;
            let sphere_r = rng.gen::<f32>() * 20.;
            let sphere = Sphere::new(sphere_center, sphere_r);

            let from = loop {
                let candidate = rng.gen::<Vec3>() * 100. - 50.;
                if candidate.distance_squared(sphere_center) > sphere_r * sphere_r + 0.0001 {
                    break candidate;
                }
            };

            let target = super::random_look_in_sphere(from, sphere_center, sphere_r, &mut rng);
            let ray = Ray {
                origin: from,
                dir: (target - from).normalize(),
            };

            assert!(sphere.t_of_intersection_point_with_ray(&ray).is_some(),);
        }
    }

    #[test]
    fn random_direction_toward_triangle() {
        let mut rng = SmallRng::seed_from_u64(0);
        for _ in 0..10 {
            let a = rng.gen::<Vec3>() * 100. - 50.;
            let b = rng.gen::<Vec3>() * 100. - 50.;
            let c = rng.gen::<Vec3>() * 100. - 50.;
            let tri = Triangle::new((a, b, c));

            let dir = super::random_direction_toward_triangle(a, b, c, &mut rng);
            let ray = Ray {
                origin: Vec3::ZERO,
                dir,
            };

            let t = tri.t_of_intersection_point_with_ray(&ray);
            assert!(!t.is_nan());
            let xyz = ray.point_from_t(t);
            let uv = tri.uv_of_point(xyz);
            assert!(Triangle::uv_is_inside(uv));
        }
    }
}
