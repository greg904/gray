use glam::Vec3;

pub struct Sphere {
    center: Vec3,
    radius: f32,
    radius_sqr: f32,
}

impl Sphere {
    pub fn new(center: Vec3, radius: f32) -> Self {
        Self {
            center,
            radius,
            radius_sqr: radius * radius,
        }
    }

    pub fn t_of_intersection_point_with_ray(&self, ray: &crate::Ray) -> f32 {
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
        let origin_to_center = ray.origin - self.center;
        let half_b = ray.dir.dot(origin_to_center);
        let half_b_sqr = half_b * half_b;
        let c = origin_to_center.length_squared() - self.radius_sqr;
        let quarter_discriminant = half_b_sqr - c;
        if quarter_discriminant < 0. {
            return f32::NAN;
        }
        let quarter_discriminant_sqrt = quarter_discriminant.sqrt();
        // If both solutions are negative, the sphere is behind the ray.
        // If one is positive and the other is negative, the ray origin is inside the sphere.
        // In that case, we do not want to count the intersection because our spheres are only
        // made up of the exterior surface. This allows shooting a ray from a point on the
        // sphere without caring about intersecting that same sphere again without convoluted
        // tricks.
        // If both are positive, then we take the smallest which is the one that corresponds
        // to the first intersection point with the sphere.
        let t_0 = -half_b - quarter_discriminant_sqrt;
        if t_0 < 0. {
            return f32::NAN;
        }
        t_0
    }

    pub fn center(&self) -> &Vec3 {
        &self.center
    }

    pub fn center_mut(&mut self) -> &mut Vec3 {
        &mut self.center
    }

    pub fn radius(&self) -> f32 {
        self.radius
    }
}
