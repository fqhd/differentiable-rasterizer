use std::f32;

pub use nalgebra::Vector2;

use crate::math;

pub struct Bezier {
    a: Vector2<f32>,
    b: Vector2<f32>,
    c: Vector2<f32>,
}

impl Bezier {
    pub fn new(a: Vector2<f32>, b: Vector2<f32>, c: Vector2<f32>) -> Bezier {
        Bezier { a, b, c }
    }

    // B(t)
    fn curve(&self, t: f32) -> Vector2<f32> {
        let a = self.a;
        let b = self.b;
        let c = self.c;

        a - 2.0 * t * a + a * t.powf(2.0) + 2.0 * b * t - 2.0 * b * t.powf(2.0) + c * t.powf(2.0)
    }

    // D(t)
    fn distance(&self, p: Vector2<f32>, t: f32) -> f32 {
        math::distance(p, self.curve(t))
    }

    fn d_distance(&self, x: f32, y: f32, t: f32) -> f32 {
        let s = self.s(x, y, t);
        let s_prime = self.s_prime(x, y, t);

        s_prime / (2.0 * s.sqrt())
    }

    fn s(&self, x: f32, y: f32, t: f32) -> f32 {
        (x - self.curve(t).x).powf(2.0) + (y - self.curve(t).y).powf(2.0)
    }

    fn s_prime(&self, x: f32, y: f32, t: f32) -> f32 {
        2.0 * (x - self.curve(t).x) * self.d_curve(t).x
            + 2.0 * (y - self.curve(t).y) * self.d_curve(t).y
    }

    fn d_curve(&self, t: f32) -> Vector2<f32> {
        let a = &self.a;
        let b = &self.b;
        let c = &self.c;

        -2.0 * a + 2.0 * a * t + 2.0 * b - 4.0 * b * t + 2.0 * c * t
    }

    fn a(&self) -> f32 {
        let y0 = self.a.y;
        let y1 = self.b.y;
        let y2 = self.c.y;
        let x0 = self.a.x;
        let x1 = self.b.x;
        let x2 = self.c.x;

        -4.0 * y0 * y0 + 16.0 * y0 * y1 - 8.0 * y0 * y2 - 16.0 * y1 * y1 + 16.0 * y1 * y2
            - 4.0 * y2 * y2
            - 4.0 * x0 * x0
            + 16.0 * x0 * x1
            - 8.0 * x0 * x2
            - 16.0 * x1 * x1
            + 16.0 * x1 * x2
            - 4.0 * x2 * x2
    }

    fn b(&self) -> f32 {
        let y0 = self.a.y;
        let y1 = self.b.y;
        let y2 = self.c.y;
        let x0 = self.a.x;
        let x1 = self.b.x;
        let x2 = self.c.x;

        12.0 * y0 * y0 - 36.0 * y0 * y1 + 12.0 * y0 * y2 + 24.0 * y1 * y1 - 12.0 * y1 * y2
            + 12.0 * x0 * x0
            - 36.0 * x0 * x1
            + 12.0 * x0 * x2
            + 24.0 * x1 * x1
            - 12.0 * x1 * x2
    }

    fn c(&self, x: f32, y: f32) -> f32 {
        let y0 = self.a.y;
        let y1 = self.b.y;
        let y2 = self.c.y;
        let x0 = self.a.x;
        let x1 = self.b.x;
        let x2 = self.c.x;

        -12.0 * y0 * y0 + 24.0 * y0 * y1 - 4.0 * y0 * y2 + 4.0 * y0 * y
            - 8.0 * y1 * y1
            - 8.0 * y1 * y
            + 4.0 * y2 * y
            - 12.0 * x0 * x0
            + 24.0 * x0 * x1
            - 4.0 * x0 * x2
            + 4.0 * x0 * x
            - 8.0 * x1 * x1
            - 8.0 * x1 * x
            + 4.0 * x2 * x
    }

    fn d(&self, x: f32, y: f32) -> f32 {
        let y0 = self.a.y;
        let y1 = self.b.y;
        let x0 = self.a.x;
        let x1 = self.b.x;

        4.0 * y0 * y0 - 4.0 * y0 * y1 - 4.0 * y0 * y + 4.0 * y1 * y + 4.0 * x0 * x0
            - 4.0 * x0 * x1
            - 4.0 * x0 * x
            + 4.0 * x1 * x
    }

    fn da_dy0(&self) -> f32 {
        -8.0 * self.a.y + 16.0 * self.b.y - 8.0 * self.c.y
    }

    fn da_dy1(&self) -> f32 {
        16.0 * self.a.y + 16.0 * self.c.y - 32.0 * self.b.y
    }

    fn da_dy2(&self) -> f32 {
        -8.0 * self.a.y + 16.0 * self.b.y - 8.0 * self.c.y
    }

    fn da_dx0(&self) -> f32 {
        16.0 * self.b.x - 8.0 * self.a.x - 8.0 * self.c.x
    }

    fn da_dx1(&self) -> f32 {
        16.0 * self.a.x - 32.0 * self.b.x + 16.0 * self.c.x
    }

    fn da_dx2(&self) -> f32 {
        -8.0 * self.a.x + 16.0 * self.b.x - 8.0 * self.c.x
    }

    fn db_dy0(&self) -> f32 {
        24.0 * self.a.y - 36.0 * self.b.y + 12.0 * self.c.y
    }

    fn db_dy1(&self) -> f32 {
        -36.0 * self.a.y + 48.0 * self.b.y - 12.0 * self.c.y
    }

    fn db_dy2(&self) -> f32 {
        12.0 * self.a.y - 12.0 * self.b.y
    }

    fn db_dx0(&self) -> f32 {
        24.0 * self.a.x - 36.0 * self.b.x + 12.0 * self.c.x
    }

    fn db_dx1(&self) -> f32 {
        -36.0 * self.a.x + 48.0 * self.b.x - 12.0 * self.c.x
    }

    fn db_dx2(&self) -> f32 {
        12.0 * self.a.x - 12.0 * self.b.x
    }

    fn dc_dy0(&self, y: f32) -> f32 {
        -24.0 * self.a.y + 24.0 * self.b.y - 4.0 * self.c.y + 4.0 * y
    }

    fn dc_dy1(&self, y: f32) -> f32 {
        24.0 * self.a.y - 16.0 * self.b.y - 8.0 * y
    }

    fn dc_dy2(&self, y: f32) -> f32 {
        -4.0 * self.a.y + 4.0 * y
    }

    fn dc_dx0(&self, x: f32) -> f32 {
        -24.0 * self.a.x + 24.0 * self.b.x - 4.0 * self.c.x + 4.0 * x
    }

    fn dc_dx1(&self, x: f32) -> f32 {
        24.0 * self.a.x - 16.0 * self.b.x - 8.0 * x
    }

    fn dc_dx2(&self, x: f32) -> f32 {
        -4.0 * self.a.x + 4.0 * x
    }

    fn dd_dy0(&self, y: f32) -> f32 {
        8.0 * self.a.y - 4.0 * self.b.y - 4.0 * y
    }

    fn dd_dy1(&self, y: f32) -> f32 {
        -4.0 * self.a.y + 4.0 * y
    }

    fn dd_dy2(&self) -> f32 {
        0.0
    }

    fn dd_dx0(&self, x: f32) -> f32 {
        8.0 * self.a.x - 4.0 * self.b.x - 4.0 * x
    }

    fn dd_dx1(&self, x: f32) -> f32 {
        -4.0 * self.a.x + 4.0 * x
    }

    fn dd_dx2(&self) -> f32 {
        0.0
    }

    pub fn forward(&self, x: f32, y: f32) -> f32 {
        let roots = math::solve_cubic(self.a(), self.b(), self.c(x, y), self.d(x, y));

        let mut t_values = vec![0.0, 1.0];
        for t in roots {
            t_values.push(t.0);
        }

        let mut min_distance = f32::MAX;
        for t in t_values {
            if (0.0..=1.0).contains(&t) {
                let d = self.distance(Vector2::new(x, y), t);
                if d < min_distance {
                    min_distance = d;
                }
            }
        }

        math::p_sigmoid(min_distance, 0.01, 2000.0)
    }

    pub fn backward(&mut self, x: f32, y: f32, y_hat: f32, target: f32) {}

    pub fn zero_grad(&mut self) {}

    pub fn step(&mut self, width: u32, lr: f32) {}
}
