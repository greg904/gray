use std::f32;

use glam::Vec3A;

#[derive(Copy, Clone)]
pub struct Light {
    pos: Vec3A,
    power_over_4pi: Vec3A,
}

impl Light {
    pub fn new(pos: Vec3A, power: Vec3A) -> Self {
        Self {
            pos,
            power_over_4pi: power / (4. * f32::consts::PI),
        }
    }

    pub fn pos(&self) -> &Vec3A {
        &self.pos
    }

    pub fn power_over_4pi(&self) -> &Vec3A {
        &self.power_over_4pi
    }
}
