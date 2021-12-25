use glam::DMat2;
use glam::DVec2;
use glam::DVec3;

pub struct Triangle {
    points: (DVec3, DVec3, DVec3),
    normal: DVec3,
    s_xyz: DVec3,
    t_xyz: DVec3,
    uv_to_st: DMat2,
}

impl Triangle {
    pub fn new(points: (DVec3, DVec3, DVec3)) -> Self {
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
        }
    }

    /// Returns the `t` parameter of the point of intersection between the ray and the plane that
    /// contains the triangle, or `NaN` if there is none. Note that although the intersection
    /// point is in the plane that contains the triangle, it could still be outside the triangle.
    pub fn t_of_intersection_point_with_ray(&self, ray: &crate::Ray) -> f64 {
        -((ray.origin - self.points.0).dot(self.normal)) / ray.dir.dot(self.normal)
    }

    /// Returns the UV coordinates of a point in space, assuming that the point is in the plane
    /// that contains the triangle.
    pub fn uv_of_point(&self, point_xyz: DVec3) -> DVec2 {
        // Get the coordinates of the point in the orthonormal basis.
        let point_st = DVec2::new(
            (point_xyz - self.points.0).dot(self.s_xyz),
            (point_xyz - self.points.0).dot(self.t_xyz),
        );
        // Use the change of basis matrix.
        self.uv_to_st * point_st
    }

    /// Returns whether the given UV coordinates are inside the triangle or not.
    pub fn uv_is_inside(uv: DVec2) -> bool {
        uv.x >= 0. && uv.y >= 0. && uv.x + uv.y <= 1.
    }

    /// Converts UV coordinates to world coordinates.
    pub fn point_from_uv(&self, uv: DVec2) -> DVec3 {
        self.points.0
            + uv.x * (self.points.1 - self.points.0)
            + uv.y * (self.points.2 - self.points.0)
    }

    pub fn normal(&self) -> DVec3 {
        self.normal
    }

    pub fn a(&self) -> DVec3 {
        self.points.0
    }

    pub fn b(&self) -> DVec3 {
        self.points.0
    }

    pub fn c(&self) -> DVec3 {
        self.points.0
    }
}
