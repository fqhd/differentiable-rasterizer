pub struct Circle {
    pub x: f32,
    pub y: f32,
    pub r: f32,
    pub dx: f32,
    pub dy: f32,
    pub dr: f32,
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
        let d = distance(x0, self.x, y0, self.y);
        let c = p_sigmoid(d, self.r, 2000.0);
        c
    }

    pub fn backward(&mut self, x0: f32, y0: f32, y_hat: f32, target: f32) {
        let dsdx = d_p_sigmoid(distance(x0, self.x, y0, self.y), self.r, 2000.0);
        let dddx = dx_distance(x0, self.x, y0, self.y);
        let dddy = dy_distance(x0, self.x, y0, self.y);
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

fn p_sigmoid(x: f32, offset: f32, sharpness: f32) -> f32 {
    1.0 / (1.0 + ((offset - x) * sharpness).clamp(-10.0, 10.0).exp())
}

fn d_p_sigmoid(x: f32, offset: f32, sharpness: f32) -> f32 {
    let a = ((offset - x) * sharpness).clamp(-10.0, 10.0).exp();
    let t1 = 1.0 / (1.0 + a).powf(2.0);
    t1 * a * sharpness
}

fn distance(x0: f32, x: f32, y0: f32, y: f32) -> f32 {
    ((x0 - x).powf(2.0) + (y0 - y).powf(2.0)).sqrt()
}

fn dx_distance(x0: f32, x: f32, y0: f32, y: f32) -> f32 {
    (x - x0) / distance(x0, x, y0, y)
}

fn dy_distance(x0: f32, x: f32, y0: f32, y: f32) -> f32 {
    (y - y0) / distance(x0, x, y0, y)
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
