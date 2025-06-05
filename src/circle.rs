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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::*;

    #[test]
    fn test_sigmoid() {
        assert_eq!(p_sigmoid(1.0, 0.4, 40.0), 1.0);
        assert!((p_sigmoid(0.5, 0.4, 40.0) - 0.982014).abs() < 0.00001)
    }

    #[test]
    fn test_dx_sigmoid() {
        assert_eq!(dx_p_sigmoid(0.4, 0.4, 40.0), 10.0);
        assert_eq!(dx_p_sigmoid(0.4, 0.4, 100.0), 25.0);
        assert!((dx_p_sigmoid(0.2, 0.4, 30.0) - 0.073995).abs() < 0.00001)
    }

    #[test]
    fn test_distance() {
        assert_eq!(
            distance(Vector2::new(0.0, 5.0), Vector2::new(0.0, 0.0)),
            5.0
        );
        assert!(
            (distance(Vector2::new(0.0, 5.0), Vector2::new(5.0, 0.0)) - 7.07107).abs() < 0.00001
        );
        assert!(
            (distance(Vector2::new(0.0, 5.0), Vector2::new(-10.0, 0.0)) - 11.18034).abs() < 0.00001
        );
    }

    #[test]
    fn test_dx_distance() {
        assert_eq!(
            dx_distance(Vector2::new(0.0, 5.0), Vector2::new(0.0, 0.0)),
            0.0
        );
        assert!(
            (dx_distance(Vector2::new(0.0, 5.0), Vector2::new(5.0, 0.0)) - 0.707107).abs()
                < 0.00001
        );
        assert!(
            (dx_distance(Vector2::new(0.0, 5.0), Vector2::new(-10.0, 0.0)) + 0.89443).abs()
                < 0.00001
        );
    }
}
