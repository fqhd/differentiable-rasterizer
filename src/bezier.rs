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
        let a = &self.a;
        let b = &self.b;
        let c = &self.c;

        let numerator = 2.0 * a - 2.0 * b;
        let denominator = 2.0 * a - 4.0 * b + 2.0 * c;

        let rootsd = numerator.component_div(&denominator);

        let mut t_values = Vec::new();

        if rootsd.x == rootsd.y {
            t_values.push(rootsd.x);
        }

        let rootsx = quad(a.x - 2.0 * b.x + c.x, 2.0 * b.x - 2.0 * a.x, a.x - x0);
        let rootsy = quad(a.y - 2.0 * b.y + c.y, 2.0 * b.y - 2.0 * a.y, a.y - y0);

        for t in rootsx {
            t_values.push(t);
        }

        for t in rootsy {
            t_values.push(t);
        }

        t_values.push(0.0);
        t_values.push(1.0);

        let mut min_distance = f32::MAX;
        for t in t_values {
            if t >= 0.0 && t <= 1.0 {
                let d = self.distance(x0, y0, t);
                if d < min_distance {
                    min_distance = d;
                }
            }
        }

        if min_distance < 0.05 { 0.0 } else { 1.0 }
    }

    pub fn backward(&mut self, x0: f32, y0: f32, y_hat: f32, target: f32) {}

    pub fn zero_grad(&mut self) {}

    pub fn step(&mut self, width: u32, lr: f32) {}
}

fn quad(a: f32, b: f32, c: f32) -> Vec<f32> {
    let discriminant = b.powf(2.0) - 4.0 * a * c;
    if discriminant == 0.0 {
        let root1 = (-b) / (2.0 * a);
        return vec![root1];
    } else if discriminant > 0.0 {
        let root1 = ((-b) + discriminant.sqrt()) / (2.0 * a);
        let root2 = ((-b) - discriminant.sqrt()) / (2.0 * a);
        return vec![root1, root2];
    } else {
        return vec![];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quadratic_function_test() {
        let a = 2.1;
        let b = 8.3;
        let c = 5.7;

        let roots = quad(a, b, c);

        let roots: Vec<f32> = roots.iter().map(|x| (x * 100.0).round()).collect();

        assert!(roots.contains(&-307.0) && roots.contains(&-88.0));
    }
}
