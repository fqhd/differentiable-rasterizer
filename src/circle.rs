use nalgebra::Vector2;

use crate::math;

pub struct Circle {
    x: f32,
    y: f32,
    r: f32,
    dx: f32,
    dy: f32,
    dr: f32,
}

impl Circle {
    pub fn new(x: f32, y: f32, radius: f32) -> Circle {
        Circle {
            x,
            y,
            r: radius,
            dx: 0.0,
            dy: 0.0,
            dr: 0.0,
        }
    }

    pub fn forward(&self, x0: f32, y0: f32) -> f32 {
        let d = math::distance(Vector2::new(x0, y0), Vector2::new(self.x, self.y));
        math::p_sigmoid(d, self.r, 2000.0)
    }

    pub fn backward(&mut self, x0: f32, y0: f32, y_hat: f32, target: f32) {
        let dsdx = math::dx_p_sigmoid(
            math::distance(Vector2::new(x0, y0), Vector2::new(self.x, self.y)),
            self.r,
            2000.0,
        );
        let dddx = math::dx_distance(Vector2::new(x0, y0), Vector2::new(self.x, self.y));
        let dddy = math::dy_distance(Vector2::new(x0, y0), Vector2::new(self.x, self.y));
        let dcdx = dsdx * dddx;
        let dcdy = dsdx * dddy;
        self.dx += 2.0 * (target - y_hat) * dcdx;
        self.dy += 2.0 * (target - y_hat) * dcdy;
    }

    pub fn zero_grad(&mut self) {
        self.dx = 0.0;
        self.dy = 0.0;
        self.dr = 0.0;
    }

    pub fn step(&mut self, width: u32, lr: f32) {
        self.dx /= (width * width) as f32;
        self.dy /= (width * width) as f32;
        self.dr /= (width * width) as f32;

        self.x += self.dx * lr;
        self.y += self.dy * lr;
        self.r += self.dr * lr;
    }
}
