


pub struct LinearMagnitudeManager {
    nb_iterations: usize,
    magnitude0: f32,
    magnitude1: f32,
}
impl LinearMagnitudeManager {

    pub fn new(nb_iterations: usize, magnitude0: f32, magnitude1: f32) -> LinearMagnitudeManager {
        LinearMagnitudeManager { nb_iterations, magnitude0, magnitude1 }
    }

    pub fn get_magnitude(&self, iteration: usize) -> f32 {
        let x = iteration as f32 / self.nb_iterations as f32;
        x * self.magnitude1 + (1.0-x) * self.magnitude0
    }

}


pub struct SkiMagnitudeManager {
    nb_iterations: usize,
    start_flag0: usize,
    start_flag1: usize,
    initial_magnitude: f32,
    f0_a: f32,
    f0_b: f32,
    f1_a: f32,
    f1_b: f32,
}
impl SkiMagnitudeManager {
    pub fn new(nb_iterations: usize, start_flag0: usize, start_flag1: usize, initial_magnitude: f32,
               f0_flag: usize, f0_magnitude_ratio_at_flag: f32,
               f1_flag: usize, f1_magnitude_ratio_at_flag: f32) -> SkiMagnitudeManager {

        let (f0_a, f0_b) = SkiMagnitudeManager::get_function_params(initial_magnitude, f0_flag, f0_magnitude_ratio_at_flag);
        let (f1_a, f1_b) = SkiMagnitudeManager::get_function_params(initial_magnitude, f1_flag, f1_magnitude_ratio_at_flag);
        SkiMagnitudeManager {
            nb_iterations,
            start_flag0,
            start_flag1,
            initial_magnitude,
            f0_a, f0_b,
            f1_a, f1_b,
        }
    }

    pub fn get_magnitude(&self, iteration: usize, mut iterations_since_change: usize) -> f32 {
        let x = iteration as f32 / self.nb_iterations as f32;
        let start_flag = (x * self.start_flag0 as f32 + (1.0-x) * self.start_flag1 as f32) as usize;
        if iterations_since_change >= start_flag {
            iterations_since_change -= start_flag;
            let f_x = iterations_since_change as f32;

            let f0 = 1.0 / (self.f0_a*f_x + self.f0_b);
            let f1 = 1.0 / (self.f1_a*f_x + self.f1_b);
            x * f1 + (1.0-x) * f0
        } else {
            self.initial_magnitude
        }
    }

    fn get_function_params(initial_magnitude: f32, flag: usize, magnitude_ratio_at_flag: f32) -> (f32, f32) {
        let flag = flag as f32;
        let a = (1.0 - magnitude_ratio_at_flag) / (initial_magnitude*flag*magnitude_ratio_at_flag);
        let b = 1.0 / initial_magnitude;
        (a,b)
        // f(x) = 1/(ax+b)
    }

}


pub struct SkiMagnitudeManagerV2 {
    initial_magnitude: f32,
    count_div: usize,
    count_add_bad_iter: usize,
    count_sub_good_iter: usize,
    last_error: f32,
    count: usize,
}
impl SkiMagnitudeManagerV2 {

    pub fn new(initial_magnitude: f32, count_div: usize, count_add_bad_iter: usize, count_sub_good_iter: usize,) -> SkiMagnitudeManagerV2 {
        SkiMagnitudeManagerV2 {
            initial_magnitude,
            count_div,
            count_add_bad_iter,
            count_sub_good_iter,
            last_error: 9999.0,
            count: count_div,
        }
    }

    pub fn update(&mut self, error: f32) {
        if error > self.last_error {
            self.count += self.count_add_bad_iter;
        } else if self.count > self.count_div {
            if self.count - self.count_sub_good_iter > self.count_div {
                self.count -= self.count_sub_good_iter;
            } else {
                self.count = self.count_div;
            }
        }
        self.last_error = error;
    }

    pub fn get_magnitude(&self) -> f32 {
        self.initial_magnitude / (self.count as f32 / self.count_div as f32)
    }

}




#[cfg(test)]
mod tests {

    #[test]
    fn test() {

    }

}