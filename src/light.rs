use std::f32;

use glam::Vec3;

#[derive(Copy, Clone)]
pub struct Light {
    pos: Vec3,
    power_over_4pi: Vec3,
}

impl Light {
    pub fn new(pos: Vec3, power: Vec3) -> Self {
        Self {
            pos,
            power_over_4pi: power / (4. * f32::consts::PI),
        }
    }

    pub fn pos(&self) -> &Vec3 {
        &self.pos
    }

    pub fn power_over_4pi(&self) -> &Vec3 {
        &self.power_over_4pi
    }
}
