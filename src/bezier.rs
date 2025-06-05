use std::f32;

pub use nalgebra::Vector2;

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
        let a = &self.a;
        let b = &self.b;
        let c = &self.c;

        a - 2.0 * t * a + a * t.powf(2.0) + 2.0 * b * t - 2.0 * b * t.powf(2.0) + c * t.powf(2.0)
    }

    // D(t)
    fn distance(&self, x: f32, y: f32, t: f32) -> f32 {
        ((x - self.curve(t).x).powf(2.0) + (y - self.curve(t).y).powf(2.0)).sqrt()
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
        let y0 = &self.a.y;
        let y1 = &self.b.y;
        let y2 = &self.c.y;
        let x0 = &self.a.x;
        let x1 = &self.b.x;
        let x2 = &self.c.x;

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
        let y0 = &self.a.y;
        let y1 = &self.b.y;
        let y2 = &self.c.y;
        let x0 = &self.a.x;
        let x1 = &self.b.x;
        let x2 = &self.c.x;

        12.0 * y0 * y0 - 36.0 * y0 * y1 + 12.0 * y0 * y2 + 24.0 * y1 * y1 - 12.0 * y1 * y2
            + 12.0 * x0 * x0
            - 36.0 * x0 * x1
            + 12.0 * x0 * x2
            + 24.0 * x1 * x1
            - 12.0 * x1 * x2
    }

    fn c(&self, x: f32, y: f32) -> f32 {
        let y0 = &self.a.y;
        let y1 = &self.b.y;
        let y2 = &self.c.y;
        let x0 = &self.a.x;
        let x1 = &self.b.x;
        let x2 = &self.c.x;

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
        let y0 = &self.a.y;
        let y1 = &self.b.y;
        let x0 = &self.a.x;
        let x1 = &self.b.x;

        4.0 * y0 * y0 - 4.0 * y0 * y1 - 4.0 * y0 * y + 4.0 * y1 * y + 4.0 * x0 * x0
            - 4.0 * x0 * x1
            - 4.0 * x0 * x
            + 4.0 * x1 * x
    }

    pub fn forward(&self, x: f32, y: f32) -> f32 {
        let roots = solve_cubic(self.a(), self.b(), self.c(x, y), self.d(x, y));

        let mut t_values = vec![0.0, 1.0];
        for t in roots {
            t_values.push(t);
        }

        let mut min_distance = f32::MAX;
        for t in t_values {
            if t >= 0.0 && t <= 1.0 {
                let d = self.distance(x, y, t);
                if d < min_distance {
                    min_distance = d;
                }
            }
        }

        p_sigmoid(min_distance, 0.01, 2000.0)
    }

    pub fn backward(&mut self, x: f32, y: f32, y_hat: f32, target: f32) {}

    pub fn zero_grad(&mut self) {}

    pub fn step(&mut self, width: u32, lr: f32) {}
}

fn p_sigmoid(x: f32, offset: f32, sharpness: f32) -> f32 {
    1.0 / (1.0 + ((offset - x) * sharpness).exp())
}

use std::f32::consts::PI;

pub fn solve_cubic(a: f32, b: f32, c: f32, d: f32) -> Vec<f32> {
    if a == 0.0 {
        panic!("Coefficient a cannot be zero for a cubic equation");
    }

    // Convert to depressed cubic: t^3 + pt + q = 0
    let a_inv = 1.0 / a;
    let b_over_3a = b * a_inv / 3.0;

    let p = (3.0 * a * c - b * b) / (3.0 * a * a);
    let q = (2.0 * b * b * b - 9.0 * a * b * c + 27.0 * a * a * d) / (27.0 * a * a * a);

    let discriminant = (q / 2.0).powi(2) + (p / 3.0).powi(3);
    let mut roots = Vec::new();

    if discriminant > 0.0 {
        // One real root
        let sqrt_disc = discriminant.sqrt();
        let u = (-q / 2.0 + sqrt_disc).cbrt();
        let v = (-q / 2.0 - sqrt_disc).cbrt();
        let t = u + v;
        roots.push(t - b_over_3a);
    } else if discriminant == 0.0 {
        // Triple or double root
        let u = (-q / 2.0).cbrt();
        roots.push(2.0 * u - b_over_3a);
        roots.push(-u - b_over_3a);
    } else {
        // Three real roots
        let r = (-p / 3.0).sqrt().powi(3);
        let phi = (-(q / 2.0) / r).acos();
        let two_sqrt_p3 = 2.0 * (-p / 3.0).sqrt();

        for k in 0..3 {
            let angle = (phi + 2.0 * PI * k as f32) / 3.0;
            let t = two_sqrt_p3 * angle.cos();
            roots.push(t - b_over_3a);
        }
    }

    roots
}
