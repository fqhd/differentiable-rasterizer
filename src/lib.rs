pub struct Circle {
    pub x: f32,
    pub y: f32,
    pub radius: f32,
}

pub struct Gradient {
    pub dx: f32,
    pub dy: f32,
    pub dr: f32,
    pub loss: f32,
}

impl Circle {
    pub fn new(x: f32, y: f32, radius: f32) -> Circle {
        Circle { x, y, radius }
    }
}

fn p_sigmoid(x: f32, offset: f32, sharpness: f32) -> f32 {
    1.0 / (1.0 + ((offset - x) * sharpness).exp())
}

fn dx_p_sigmoid(x: f32, offset: f32, sharpness: f32) -> f32 {
    if x < 0.2 {
        return 0.0;
    } else if x > 0.5 {
        return 0.0;
    }
    let a = ((offset - x) * sharpness).exp();
    let t1 = 1.0 / (1.0 + a).powf(2.0);
    t1 * a * sharpness
}

fn distance(x0: f32, x: f32, y0: f32, y: f32) -> f32 {
    ((x0 - x).powf(2.0) + (y0 - y).powf(2.0)).sqrt()
}

fn dx_distance(x0: f32, x: f32, y0: f32, y: f32) -> f32 {
    -(x0 - x) / distance(x0, x, y0, y)
}

fn dy_distance(x0: f32, x: f32, y0: f32, y: f32) -> f32 {
    -(y0 - y) / distance(x0, x, y0, y)
}

pub fn rasterize(circle: &Circle, width: u32) -> Vec<f32> {
    let mut image = vec![1.0; (width * width) as usize];

    for j in 0..width {
        for i in 0..width {
            let index = j * width + i;
            let y = (j as f32) / (width as f32);
            let x = (i as f32) / (width as f32);
            let d = distance(x, circle.x, y, circle.y);
            let c = p_sigmoid(d, circle.radius, 2000.0);
            image[index as usize] = c;
        }
    }

    image
}

pub fn compute_gradients(
    circle: &Circle,
    width: u32,
    raster: &Vec<f32>,
    target: &Vec<f32>,
) -> Gradient {
    let mut dldx = 0.0;
    let mut loss = 0.0;

    for j in 0..width {
        for i in 0..width {
            let index = j * width + i;
            let c = raster[index as usize];
            let label = target[index as usize];
            let y = (j as f32) / (width as f32);
            let x = (i as f32) / (width as f32);
            loss += (c - label).powf(2.0);
            let dsdx = dx_p_sigmoid(distance(x, circle.x, y, circle.y), circle.radius, 2000.0);
            let dddx = dx_distance(x, circle.x, y, circle.y);
            let dcdx = dsdx * dddx;
            dldx += 2.0 * (label - c) * dcdx;
        }
    }

    dldx /= (width * width) as f32;
    loss /= (width * width) as f32;

    Gradient {
        dx: dldx,
        dy: 0.0,
        dr: 0.0,
        loss,
    }
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
    fn test_dx_sigmoid() {
        assert_eq!(dx_p_sigmoid(0.4, 0.4, 40.0), 10.0);
        assert_eq!(dx_p_sigmoid(0.4, 0.4, 100.0), 25.0);
        assert!((dx_p_sigmoid(0.2, 0.4, 30.0) - 0.073995).abs() < 0.00001)
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
