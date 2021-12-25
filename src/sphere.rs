use glam::DVec3;

pub struct Sphere {
    center: DVec3,
    radius: f64,
    radius_sqr: f64,
}

impl Sphere {
    pub fn new(center: DVec3, radius: f64) -> Self {
        Self {
            center,
            radius,
            radius_sqr: radius * radius,
        }
    }

    pub fn t_of_intersection_point_with_ray(&self, ray: &crate::Ray) -> Option<f64> {
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

    pub fn center(&self) -> DVec3 {
        self.center
    }

    pub fn center_mut(&mut self) -> &mut DVec3 {
        &mut self.center
    }

    pub fn radius(&self) -> f64 {
        self.radius
    }
}
