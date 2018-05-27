

use CudaHandleHolder;
use GetParams;

pub trait ErrorCalculator<T: GetParams> {
    fn compute_error(&mut self, trainable: &T, cuda: &mut CudaHandleHolder) -> f32;
}







