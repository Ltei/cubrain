
use cumath::*;
use CudaHandleHolder;


pub trait VectorUtils {
    fn clone_randomized_from(&mut self, cuda: &mut CudaHandleHolder, to_clone_params: &CuVectorDeref<f32>, magnitude: f32);
    fn clone_weighted_from(&mut self, cuda: &mut CudaHandleHolder, to_clone_params: &[&CuVectorDeref<f32>], weights: &[f32]);
}

impl VectorUtils for CuVectorDeref<f32> {
    fn clone_randomized_from(&mut self, cuda: &mut CudaHandleHolder, to_clone_params: &CuVectorDeref<f32>, magnitude: f32) {
        assert_eq!(self.len(), to_clone_params.len());
        cuda.curand.generate_uniform_range(self, -magnitude, magnitude, &DEFAULT_STREAM); //TODO Choose generation
        self.add(to_clone_params, &DEFAULT_STREAM);
    }
    fn clone_weighted_from(&mut self, cuda: &mut CudaHandleHolder, to_clone_params: &[&CuVectorDeref<f32>], weights: &[f32]) {
        assert_eq!(to_clone_params.len(), weights.len());
        self.init(0.0, &DEFAULT_STREAM);
        for i in 0..to_clone_params.len() {
            cuda.cublas.axpy(weights[i], to_clone_params[i], self);
        }
    }
}