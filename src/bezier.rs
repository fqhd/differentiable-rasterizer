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
    fn distance(&self, x0: f32, y0: f32, t: f32) -> f32 {
        ((x0 - self.curve(t).x).powf(2.0) + (y0 - self.curve(t).y).powf(2.0)).sqrt()
    }

    fn d_distance(&self, x0: f32, y0: f32, t: f32) -> f32 {
        let s = self.s(x0, y0, t);
        let s_prime = self.s_prime(x0, y0, t);

        s_prime / (2.0 * s.sqrt())
    }

    fn s(&self, x0: f32, y0: f32, t: f32) -> f32 {
        (x0 - self.curve(t).x).powf(2.0) + (y0 - self.curve(t).y).powf(2.0)
    }

    fn s_prime(&self, x0: f32, y0: f32, t: f32) -> f32 {
        2.0 * (x0 - self.curve(t).x) * self.d_curve(t).x
            + 2.0 * (y0 - self.curve(t).y) * self.d_curve(t).y
    }

    fn d_curve(&self, t: f32) -> Vector2<f32> {
        let a = &self.a;
        let b = &self.b;
        let c = &self.c;

        -2.0 * a + 2.0 * a * t + 2.0 * b - 4.0 * b * t + 2.0 * c * t
    }

    pub fn forward(&self, x0: f32, y0: f32) -> f32 {
        let a = &self.a.y;
        let b = &self.b.y;
        let c = &self.c.y;
        let d = &self.a.x;
        let e = &self.b.x;
        let f = &self.c.x;
        let y = y0;
        let x = x0;

        let t3 = -4.0 * a * a + 16.0 * a * b - 8.0 * a * c - 16.0 * b * b + 16.0 * b * c
            - 4.0 * c * c
            - 4.0 * d * d
            + 16.0 * d * e
            - 8.0 * d * f
            - 16.0 * e * e
            + 16.0 * e * f
            - 4.0 * f * f;

        // Coefficient of t^2
        let t2 = 12.0 * a * a - 36.0 * a * b + 12.0 * a * c + 24.0 * b * b - 12.0 * b * c
            + 12.0 * d * d
            - 36.0 * d * e
            + 12.0 * d * f
            + 24.0 * e * e
            - 12.0 * e * f;

        // Coefficient of t^1
        let t1 =
            -12.0 * a * a + 24.0 * a * b - 4.0 * a * c + 4.0 * a * y - 8.0 * b * b - 8.0 * b * y
                + 4.0 * c * y
                - 12.0 * d * d
                + 24.0 * d * e
                - 4.0 * d * f
                + 4.0 * d * x
                - 8.0 * e * e
                - 8.0 * e * x
                + 4.0 * f * x;

        // Constant term (t^0)
        let t0 = 4.0 * a * a - 4.0 * a * b - 4.0 * a * y + 4.0 * b * y + 4.0 * d * d
            - 4.0 * d * e
            - 4.0 * d * x
            + 4.0 * e * x;

        let roots = solve_cubic(t3, t2, t1, t0);

        let mut t_values = vec![0.0, 1.0];
        for t in roots {
            t_values.push(t);
        }

        let mut min_distance = f32::MAX;
        for t in t_values {
            if t >= 0.0 && t <= 1.0 {
                let d = self.distance(x0, y0, t);
                if d < min_distance {
                    min_distance = d;
                }
            }
        }

        p_sigmoid(min_distance, 0.01, 2000.0)
    }

    pub fn backward(&mut self, x0: f32, y0: f32, y_hat: f32, target: f32) {}

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
