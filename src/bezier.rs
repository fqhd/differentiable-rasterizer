use std::f32;

pub use nalgebra::Vector2;

use crate::math;

pub struct Bezier {
    a: Vector2<f32>,
    b: Vector2<f32>,
    c: Vector2<f32>,

    pub da: Vector2<f32>,
    db: Vector2<f32>,
    dc: Vector2<f32>,
}

impl Bezier {
    pub fn new(a: Vector2<f32>, b: Vector2<f32>, c: Vector2<f32>) -> Bezier {
        Bezier {
            a,
            b,
            c,
            da: Vector2::zeros(),
            db: Vector2::zeros(),
            dc: Vector2::zeros(),
        }
    }

    // B(t)
    fn curve(&self, t: f32) -> Vector2<f32> {
        let a = self.a;
        let b = self.b;
        let c = self.c;

        a - 2.0 * t * a + a * t.powf(2.0) + 2.0 * b * t - 2.0 * b * t.powf(2.0) + c * t.powf(2.0)
    }

    // Could optimize this by only running curve once
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
        let mut roots = math::solve_cubic(self.a(), self.b(), self.c(x, y), self.d(x, y));

        roots.push((1.0, 0));
        roots.push((0.0, 0));

        let mut min_distance = (f32::MAX, 0);
        for t in roots {
            if (0.0..=1.0).contains(&t.0) {
                let d = math::squared_distance(Vector2::new(x, y), self.curve(t.0));
                if d < min_distance.0 {
                    min_distance.0 = d;
                    min_distance.1 = t.1;
                }
            }
        }

        math::p_sigmoid(min_distance.0, 0.01, 1000.0)
    }

    pub fn forward_with_gradients(&mut self, x: f32, y: f32, target: f32) -> (f32, f32) {
        let a = self.a();
        let b = self.b();
        let c = self.c(x, y);
        let d = self.d(x, y);

        let mut t_values = math::solve_cubic(a, b, c, d);

        t_values.push((0.0, 6));
        t_values.push((1.0, 7));

        let mut min_distance = (f32::MAX, 0);
        let mut min_t = f32::MAX;
        for t in t_values {
            if (0.0..=1.0).contains(&t.0) {
                let d = math::squared_distance(Vector2::new(x, y), self.curve(t.0));
                if d < min_distance.0 {
                    min_distance.0 = d;
                    min_distance.1 = t.1;
                    min_t = t.0;
                }
            }
        }

        let colour = math::p_sigmoid(min_distance.0, 0.01, 1000.0);
        let loss = (colour - target).powf(2.0);

        if min_distance.1 == 7 || min_distance.1 == 6 {
            // If tag is 6 or 7 then derivative is 0
            return (colour, loss);
        }

        let distance_derivative = self.s_prime(x, y, min_t); //
        let sigmoid_derivative = math::dx_p_sigmoid(min_distance.0, 0.01, 1000.0);

        let combined = sigmoid_derivative * distance_derivative;

        let mut dx0 = combined
            * math::d_solve_cubic(
                a,
                b,
                c,
                d,
                min_distance.1,
                (self.da_dx0(), self.db_dx0(), self.dc_dx0(x), self.dd_dx0(x)),
            );

        let dx1 = combined
            * math::d_solve_cubic(
                a,
                b,
                c,
                d,
                min_distance.1,
                (self.da_dx1(), self.db_dx1(), self.dc_dx1(x), self.dd_dx1(x)),
            );

        let dx2 = combined
            * math::d_solve_cubic(
                a,
                b,
                c,
                d,
                min_distance.1,
                (self.da_dx2(), self.db_dx2(), self.dc_dx2(x), self.dd_dx2()),
            );

        let dy0 = combined
            * math::d_solve_cubic(
                a,
                b,
                c,
                d,
                min_distance.1,
                (self.da_dy0(), self.db_dy0(), self.dc_dy0(y), self.dd_dy0(y)),
            );

        let dy1 = combined
            * math::d_solve_cubic(
                a,
                b,
                c,
                d,
                min_distance.1,
                (self.da_dy1(), self.db_dy1(), self.dc_dy1(y), self.dd_dy1(y)),
            );

        let dy2 = combined
            * math::d_solve_cubic(
                a,
                b,
                c,
                d,
                min_distance.1,
                (self.da_dy2(), self.db_dy2(), self.dc_dy2(y), self.dd_dy2()),
            );

        if !dx0.is_finite() {
            dx0 = 0.0;
        }

        let dloss = 2.0 * (colour - target);

        self.da.x += dloss * dx0;
        self.da.y += dloss * dy0;
        self.db.x += dloss * dx1;
        self.db.y += dloss * dy1;
        self.dc.x += dloss * dx2;
        self.dc.y += dloss * dy2;

        (colour, loss)
    }

    pub fn zero_grad(&mut self) {
        self.da.x = 0.0;
        self.da.y = 0.0;
        self.db.x = 0.0;
        self.db.y = 0.0;
        self.dc.x = 0.0;
        self.dc.y = 0.0;
    }

    pub fn step(&mut self, width: u32, lr: f32) {
        self.da /= (width * width) as f32;
        self.db /= (width * width) as f32;
        self.dc /= (width * width) as f32;

        self.a += self.da * lr;
        self.b += self.db * lr;
        self.c += self.dc * lr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dady0() {
        let mut curve = Bezier::new(
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(1.0, 1.0),
        );

        let dy = 1e-5;
        let y1 = curve.a();
        curve.a.y += dy;
        let y2 = curve.a();

        let approx = (y2 - y1) / dy;
        let truth = curve.da_dy0();

        assert!(
            (approx - truth).abs() < 0.1,
            "Got different values curve: {}, approx: {}",
            truth,
            approx
        );
    }

    #[test]
    fn test_dady1() {
        let mut curve = Bezier::new(
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(1.0, 1.0),
        );

        let dy = 1e-5;
        let y1 = curve.a();
        curve.b.y += dy;
        let y2 = curve.a();

        let approx = (y2 - y1) / dy;
        let truth = curve.da_dy1();

        assert!(
            (approx - truth).abs() < 0.1,
            "Got different values curve: {}, approx: {}",
            truth,
            approx
        );
    }

    #[test]
    fn test_dady2() {
        let mut curve = Bezier::new(
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(1.0, 1.0),
        );

        let dy = 1e-5;
        let y1 = curve.a();
        curve.c.y += dy;
        let y2 = curve.a();

        let approx = (y2 - y1) / dy;
        let truth = curve.da_dy2();

        assert!(
            (approx - truth).abs() < 0.1,
            "Got different values curve: {}, approx: {}",
            truth,
            approx
        );
    }

    #[test]
    fn test_dadx0() {
        let mut curve = Bezier::new(
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(1.0, 1.0),
        );

        let dx = 1e-5;
        let y1 = curve.a();
        curve.a.x += dx;
        let y2 = curve.a();

        let approx = (y2 - y1) / dx;
        let truth = curve.da_dx0();

        assert!(
            (approx - truth).abs() < 0.1,
            "Got different values curve: {}, approx: {}",
            truth,
            approx
        );
    }

    #[test]
    fn test_dadx1() {
        let mut curve = Bezier::new(
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(1.0, 1.0),
        );

        let dx = 1e-5;
        let y1 = curve.a();
        curve.b.x += dx;
        let y2 = curve.a();

        let approx = (y2 - y1) / dx;
        let truth = curve.da_dx1();

        assert!(
            (approx - truth).abs() < 0.1,
            "Got different values curve: {}, approx: {}",
            truth,
            approx
        );
    }

    #[test]
    fn test_dadx2() {
        let mut curve = Bezier::new(
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(1.0, 1.0),
        );

        let dx = 1e-5;
        let y1 = curve.a();
        curve.c.x += dx;
        let y2 = curve.a();

        let approx = (y2 - y1) / dx;
        let truth = curve.da_dx2();

        assert!(
            (approx - truth).abs() < 0.1,
            "Got different values curve: {}, approx: {}",
            truth,
            approx
        );
    }

    #[test]
    fn test_dbdy0() {
        let mut curve = Bezier::new(
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(1.0, 1.0),
        );

        let dy = 1e-5;
        let y1 = curve.b();
        curve.a.y += dy;
        let y2 = curve.b();

        let approx = (y2 - y1) / dy;
        let truth = curve.db_dy0();

        assert!(
            (approx - truth).abs() < 0.1,
            "Got different values curve: {}, approx: {}",
            truth,
            approx
        );
    }

    #[test]
    fn test_dbdy1() {
        let mut curve = Bezier::new(
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(1.0, 1.0),
        );

        let dy = 1e-5;
        let y1 = curve.b();
        curve.b.y += dy;
        let y2 = curve.b();

        let approx = (y2 - y1) / dy;
        let truth = curve.db_dy1();

        assert!(
            (approx - truth).abs() < 0.1,
            "Got different values curve: {}, approx: {}",
            truth,
            approx
        );
    }

    #[test]
    fn test_dbdy2() {
        let mut curve = Bezier::new(
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(1.0, 1.0),
        );

        let dy = 1e-5;
        let y1 = curve.b();
        curve.c.y += dy;
        let y2 = curve.b();

        let approx = (y2 - y1) / dy;
        let truth = curve.db_dy2();

        assert!(
            (approx - truth).abs() < 0.1,
            "Got different values curve: {}, approx: {}",
            truth,
            approx
        );
    }

    #[test]
    fn test_dbdx0() {
        let mut curve = Bezier::new(
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(1.0, 1.0),
        );

        let dx = 1e-5;
        let y1 = curve.b();
        curve.a.x += dx;
        let y2 = curve.b();

        let approx = (y2 - y1) / dx;
        let truth = curve.db_dx0();

        assert!(
            (approx - truth).abs() < 0.1,
            "Got different values curve: {}, approx: {}",
            truth,
            approx
        );
    }

    #[test]
    fn test_dbdx1() {
        let mut curve = Bezier::new(
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(1.0, 1.0),
        );

        let dx = 1e-5;
        let y1 = curve.b();
        curve.b.x += dx;
        let y2 = curve.b();

        let approx = (y2 - y1) / dx;
        let truth = curve.db_dx1();

        assert!(
            (approx - truth).abs() < 0.1,
            "Got different values curve: {}, approx: {}",
            truth,
            approx
        );
    }

    #[test]
    fn test_dbdx2() {
        let mut curve = Bezier::new(
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(1.0, 1.0),
        );

        let dx = 1e-5;
        let y1 = curve.b();
        curve.c.x += dx;
        let y2 = curve.b();

        let approx = (y2 - y1) / dx;
        let truth = curve.db_dx2();

        assert!(
            (approx - truth).abs() < 0.1,
            "Got different values curve: {}, approx: {}",
            truth,
            approx
        );
    }

    #[test]
    fn test_dcdy0() {
        let mut curve = Bezier::new(
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(1.0, 1.0),
        );

        let dy = 1e-5;
        let y1 = curve.c(0.0, 0.0);
        curve.a.y += dy;
        let y2 = curve.c(0.0, 0.0);

        let approx = (y2 - y1) / dy;
        let truth = curve.dc_dy0(0.0);

        assert!(
            (approx - truth).abs() < 0.1,
            "Got different values curve: {}, approx: {}",
            truth,
            approx
        );
    }

    #[test]
    fn test_dcdy1() {
        let mut curve = Bezier::new(
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(1.0, 1.0),
        );

        let dy = 1e-5;
        let y1 = curve.c(0.0, 0.0);
        curve.b.y += dy;
        let y2 = curve.c(0.0, 0.0);

        let approx = (y2 - y1) / dy;
        let truth = curve.dc_dy1(0.0);

        assert!(
            (approx - truth).abs() < 0.1,
            "Got different values curve: {}, approx: {}",
            truth,
            approx
        );
    }

    #[test]
    fn test_dcdy2() {
        let mut curve = Bezier::new(
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(1.0, 1.0),
        );

        let dy = 1e-5;
        let y1 = curve.c(0.0, 0.0);
        curve.c.y += dy;
        let y2 = curve.c(0.0, 0.0);

        let approx = (y2 - y1) / dy;
        let truth = curve.dc_dy2(0.0);

        assert!(
            (approx - truth).abs() < 0.1,
            "Got different values curve: {}, approx: {}",
            truth,
            approx
        );
    }

    #[test]
    fn test_dcdx0() {
        let mut curve = Bezier::new(
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(1.0, 1.0),
        );

        let dx = 1e-5;
        let y1 = curve.c(0.0, 0.0);
        curve.a.x += dx;
        let y2 = curve.c(0.0, 0.0);

        let approx = (y2 - y1) / dx;
        let truth = curve.dc_dx0(0.0);

        assert!(
            (approx - truth).abs() < 0.1,
            "Got different values curve: {}, approx: {}",
            truth,
            approx
        );
    }

    #[test]
    fn test_dcdx1() {
        let mut curve = Bezier::new(
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(1.0, 1.0),
        );

        let dx = 1e-5;
        let y1 = curve.c(0.0, 0.0);
        curve.b.x += dx;
        let y2 = curve.c(0.0, 0.0);

        let approx = (y2 - y1) / dx;
        let truth = curve.dc_dx1(0.0);

        assert!(
            (approx - truth).abs() < 0.1,
            "Got different values curve: {}, approx: {}",
            truth,
            approx
        );
    }

    #[test]
    fn test_dcdx2() {
        let mut curve = Bezier::new(
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(1.0, 1.0),
        );

        let dx = 1e-5;
        let y1 = curve.c(0.0, 0.0);
        curve.c.x += dx;
        let y2 = curve.c(0.0, 0.0);

        let approx = (y2 - y1) / dx;
        let truth = curve.dc_dx2(0.0);

        assert!(
            (approx - truth).abs() < 0.1,
            "Got different values curve: {}, approx: {}",
            truth,
            approx
        );
    }

    #[test]
    fn test_dddy0() {
        let mut curve = Bezier::new(
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(1.0, 1.0),
        );

        let dy = 1e-5;
        let y1 = curve.d(0.0, 0.0);
        curve.a.y += dy;
        let y2 = curve.d(0.0, 0.0);

        let approx = (y2 - y1) / dy;
        let truth = curve.dd_dy0(0.0);

        assert!(
            (approx - truth).abs() < 0.1,
            "Got different values curve: {}, approx: {}",
            truth,
            approx
        );
    }

    #[test]
    fn test_dddy1() {
        let mut curve = Bezier::new(
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(1.0, 1.0),
        );

        let dy = 1e-5;
        let y1 = curve.d(0.0, 0.0);
        curve.b.y += dy;
        let y2 = curve.d(0.0, 0.0);

        let approx = (y2 - y1) / dy;
        let truth = curve.dd_dy1(0.0);

        assert!(
            (approx - truth).abs() < 0.1,
            "Got different values curve: {}, approx: {}",
            truth,
            approx
        );
    }

    #[test]
    fn test_dddy2() {
        let mut curve = Bezier::new(
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(1.0, 1.0),
        );

        let dy = 1e-5;
        let y1 = curve.d(0.0, 0.0);
        curve.c.y += dy;
        let y2 = curve.d(0.0, 0.0);

        let approx = (y2 - y1) / dy;
        let truth = curve.dd_dy2();

        assert!(
            (approx - truth).abs() < 0.1,
            "Got different values curve: {}, approx: {}",
            truth,
            approx
        );
    }

    #[test]
    fn test_dddx0() {
        let mut curve = Bezier::new(
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(1.0, 1.0),
        );

        let dx = 1e-5;
        let y1 = curve.d(0.0, 0.0);
        curve.a.x += dx;
        let y2 = curve.d(0.0, 0.0);

        let approx = (y2 - y1) / dx;
        let truth = curve.dd_dx0(0.0);

        assert!(
            (approx - truth).abs() < 0.1,
            "Got different values curve: {}, approx: {}",
            truth,
            approx
        );
    }

    #[test]
    fn test_dddx1() {
        let mut curve = Bezier::new(
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(1.0, 1.0),
        );

        let dx = 1e-5;
        let y1 = curve.d(0.0, 0.0);
        curve.b.x += dx;
        let y2 = curve.d(0.0, 0.0);

        let approx = (y2 - y1) / dx;
        let truth = curve.dd_dx1(0.0);

        assert!(
            (approx - truth).abs() < 0.1,
            "Got different values curve: {}, approx: {}",
            truth,
            approx
        );
    }

    #[test]
    fn test_dddx2() {
        let mut curve = Bezier::new(
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(1.0, 1.0),
        );

        let dx = 1e-5;
        let y1 = curve.d(0.0, 0.0);
        curve.c.x += dx;
        let y2 = curve.d(0.0, 0.0);

        let approx = (y2 - y1) / dx;
        let truth = curve.dd_dx2();

        assert!(
            (approx - truth).abs() < 0.1,
            "Got different values curve: {}, approx: {}",
            truth,
            approx
        );
    }
}
