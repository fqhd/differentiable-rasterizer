pub struct Circle {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub r: f32,
    pub g: f32,
    pub b: f32,

    pub dx: f32,
    pub dy: f32,
    pub dz: f32,
    pub dr: f32,
    pub dg: f32,
    pub db: f32,

    pub vx: f32,
    pub vy: f32,
    pub vz: f32,
    pub vr: f32,
    pub vg: f32,
    pub vb: f32,
}

impl Circle {
    pub fn new(x: f32, y: f32, z: f32, r: f32, g: f32, b: f32) -> Circle {
        Circle {
            x,
            y,
            z,
            r,
            g,
            b,
            dx: 0.0,
            dy: 0.0,
            dz: 0.0,
            dr: 0.0,
            dg: 0.0,
            db: 0.0,
            vx: 0.0,
            vy: 0.0,
            vz: 0.0,
            vr: 0.0,
            vg: 0.0,
            vb: 0.0,
        }
    }

    pub fn forward(&self, x0: f32, y0: f32) -> (f32, f32, f32) {
        let d = distance(x0, self.x, y0, self.y);
        let c = p_sigmoid(d, self.z, 2000.0);
        (c * self.r, c * self.g, c * self.b)
    }

    pub fn backward(&mut self, x0: f32, y0: f32, y_hat: (f32, f32, f32), target: (f32, f32, f32)) {
        let dsdx = d_p_sigmoid(distance(x0, self.x, y0, self.y), self.z, 2000.0);
        let dddx = dx_distance(x0, self.x, y0, self.y);
        let dddy = dy_distance(x0, self.x, y0, self.y);
        let dcdx = dsdx * dddx;
        let dcdy = dsdx * dddy;
        let dcdz = dz_p_sigmoid(distance(x0, self.x, y0, self.y), self.z, 2000.0);

        self.dx += 2.0 * (y_hat.0 - target.0) * dcdx * self.r;
        self.dy += 2.0 * (y_hat.0 - target.0) * dcdy * self.r;
        self.dz += 2.0 * (y_hat.0 - target.0) * dcdz * self.r;

        self.dx += 2.0 * (y_hat.1 - target.1) * dcdx * self.g;
        self.dy += 2.0 * (y_hat.1 - target.1) * dcdy * self.g;
        self.dz += 2.0 * (y_hat.1 - target.1) * dcdz * self.g;

        self.dx += 2.0 * (y_hat.2 - target.2) * dcdx * self.b;
        self.dy += 2.0 * (y_hat.2 - target.2) * dcdy * self.b;
        self.dz += 2.0 * (y_hat.2 - target.2) * dcdz * self.b;

        let d = distance(x0, self.x, y0, self.y);
        let c = p_sigmoid(d, self.z, 2000.0);

        self.dr += 2.0 * (y_hat.0 - target.0) * 10.0 * c;
        self.dg += 2.0 * (y_hat.1 - target.1) * 10.0 * c;
        self.db += 2.0 * (y_hat.2 - target.2) * 10.0 * c;
    }

    pub fn zero_grad(&mut self) {
        self.dx = 0.0;
        self.dy = 0.0;
        self.dz = 0.0;
        self.dr = 0.0;
        self.dg = 0.0;
        self.db = 0.0;
    }

    pub fn step(&mut self, width: u32, lr: f32, momentum: f32) {
        self.dx /= (width * width * 3) as f32;
        self.dy /= (width * width * 3) as f32;
        self.dz /= (width * width * 3) as f32;
        self.dr /= (width * width) as f32;
        self.dg /= (width * width) as f32;
        self.db /= (width * width) as f32;

        self.vx = momentum * self.vx - lr * self.dx;
        self.vy = momentum * self.vy - lr * self.dy;
        self.vz = momentum * self.vz - lr * self.dz;
        self.vr = momentum * self.vr - lr * self.dr;
        self.vg = momentum * self.vg - lr * self.dg;
        self.vb = momentum * self.vb - lr * self.db;

        self.x += self.vx;
        self.y += self.vy;
        self.z += self.vz;
        self.r += self.vr;
        self.g += self.vg;
        self.b += self.vb;
    }
}

fn p_sigmoid(x: f32, offset: f32, sharpness: f32) -> f32 {
    1.0 / (1.0 + ((x - offset) * sharpness).clamp(-10.0, 10.0).exp())
}

fn d_p_sigmoid(x: f32, offset: f32, sharpness: f32) -> f32 {
    let a = ((offset - x) * sharpness).clamp(-10.0, 10.0).exp();
    let t1 = 1.0 / (1.0 + a).powf(2.0);
    t1 * a * sharpness
}

fn dz_p_sigmoid(x: f32, offset: f32, sharpness: f32) -> f32 {
    (((offset - x) * sharpness).clamp(-10.0, 10.0).exp() * sharpness)
        / (1.0 + ((offset - x) * sharpness).clamp(-10.0, 10.0).exp()).powi(2)
}

fn distance(x0: f32, x: f32, y0: f32, y: f32) -> f32 {
    ((x0 - x).powf(2.0) + (y0 - y).powf(2.0)).sqrt()
}

fn dx_distance(x0: f32, x: f32, y0: f32, y: f32) -> f32 {
    (x0 - x) / distance(x0, x, y0, y).max(1e-10)
}

fn dy_distance(x0: f32, x: f32, y0: f32, y: f32) -> f32 {
    (y0 - y) / distance(x0, x, y0, y).max(1e-10)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert_eq!(p_sigmoid(1.0, 0.4, 40.0), 1.0);
        assert!((p_sigmoid(0.5, 0.4, 40.0) - 0.982014).abs() < 0.00001)
    }

    #[test]
    fn test_d_sigmoid() {
        assert_eq!(d_p_sigmoid(0.4, 0.4, 40.0), 10.0);
        assert_eq!(d_p_sigmoid(0.4, 0.4, 100.0), 25.0);
        assert!((d_p_sigmoid(0.2, 0.4, 30.0) - 0.073995).abs() < 0.00001)
    }

    #[test]
    fn test_distance() {
        assert_eq!(distance(0.0, 0.0, 5.0, 0.0), 5.0);
        assert!((distance(0.0, 5.0, 5.0, 0.0) - 7.07107).abs() < 0.00001);
        assert!((distance(0.0, -10.0, 5.0, 0.0) - 11.18034).abs() < 0.00001);
    }

    #[test]
    fn test_dx_distance() {
        assert_eq!(dx_distance(0.0, 0.0, 5.0, 0.0), 0.0);
        assert!((dx_distance(0.0, 5.0, 5.0, 0.0) - 0.707107).abs() < 0.00001);
        assert!((dx_distance(0.0, -10.0, 5.0, 0.0) + 0.89443).abs() < 0.00001);
    }
}
