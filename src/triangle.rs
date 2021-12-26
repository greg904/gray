use glam::Mat2;
use glam::Vec2;
use glam::Vec3A;

pub struct Triangle {
    points: (Vec3A, Vec3A, Vec3A),
    normal: Vec3A,
    s_xyz: Vec3A,
    t_xyz: Vec3A,
    uv_to_st: Mat2,
}

impl Triangle {
    pub fn new(points: (Vec3A, Vec3A, Vec3A)) -> Self {
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
        let u_st = Vec2::new((points.1 - points.0).length(), 0.);
        let v_st = Vec2::new(v_xyz.dot(s_xyz), v_xyz.dot(t_xyz));
        let st_to_uv = Mat2::from_cols(u_st, v_st);
        let uv_to_st = st_to_uv.inverse();

        Triangle {
            points,
            normal: (points.1 - points.0).cross(points.2 - points.0).normalize(),
            s_xyz,
            t_xyz,
            uv_to_st,
        }
    }

    /// Returns the `t` parameter of the point of intersection between the ray and the plane that
    /// contains the triangle if it exists, or `NaN` if there is no intersection because the ray
    /// and the plane are parallel.
    ///
    /// Note that even if the intersection point is in the plane that contains the triangle, it
    /// could still be outside the triangle, so to check if there is an intersection you should
    /// also call `uv_of_point` and `uv_is_inside`.
    pub fn t_of_intersection_point_with_ray(&self, ray: &crate::Ray) -> f32 {
        ((self.points.0 - ray.origin).dot(self.normal)) / ray.dir.dot(self.normal)
    }

    /// Returns the `t` parameter of the point of intersection between the ray and the plane that
    /// contains the triangle if ray is seeing the face of the triangle that is visible, or `NaN`
    /// if the ray is seeing the face of the triangle that is not visible or if there is no intersection
    /// because the ray and the plane are parallel.
    ///
    /// Note that even if the intersection point is in the plane that contains the triangle, it
    /// could still be outside the triangle, so to check if there is an intersection you should
    /// also call `uv_of_point` and `uv_is_inside`.
    pub fn t_of_intersection_point_with_ray_checked(&self, ray: &crate::Ray) -> f32 {
        let x = ray.dir.dot(self.normal);
        if x >= 0. {
            return f32::NAN;
        }
        ((self.points.0 - ray.origin).dot(self.normal)) / x
    }

    /// Returns the UV coordinates of a point in space, assuming that the point is in the plane
    /// that contains the triangle.
    pub fn uv_of_point(&self, p: Vec3A) -> Vec2 {
        // Get the coordinates of the point in the orthonormal basis.
        let point_xyz = p - self.points.0;
        let point_st = Vec2::new(point_xyz.dot(self.s_xyz), point_xyz.dot(self.t_xyz));
        // Use the change of basis matrix.
        self.uv_to_st * point_st
    }

    /// Returns whether the given UV coordinates are inside the triangle or not.
    pub fn uv_is_inside(uv: Vec2) -> bool {
        uv.x >= 0. && uv.y >= 0. && uv.x + uv.y <= 1.
    }

    /// Converts UV coordinates to world coordinates.
    pub fn point_from_uv(&self, uv: Vec2) -> Vec3A {
        self.points.0
            + uv.x * (self.points.1 - self.points.0)
            + uv.y * (self.points.2 - self.points.0)
    }

    pub fn normal(&self) -> Vec3A {
        self.normal
    }

    pub fn a(&self) -> Vec3A {
        self.points.0
    }

    pub fn b(&self) -> Vec3A {
        self.points.1
    }

    pub fn c(&self) -> Vec3A {
        self.points.2
    }

    pub fn barycenter(&self) -> Vec3A {
        (self.points.0 + self.points.1 + self.points.2) / 3.
    }
}
