

use std::convert::From;
use cumath::{CuVector, CuVectorMutPtr, CuVectorDeref, DEFAULT_STREAM};
use meta::Save;
use super::{ComputationDescriptor, ComputationParams};
use CubrainResult;
use CudaHandleHolder;



pub trait Computation {

    fn descriptor(&self) -> &ComputationDescriptor;
    fn params(&self) -> &CuVectorDeref<f32>;
    fn params_mut(&mut self) -> &mut CuVectorDeref<f32>;

    fn forward_inference(&self, cuda: &mut CudaHandleHolder, input: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>, workspace: &mut CuVectorDeref<f32>);
    fn forward_training(&self, cuda: &mut CudaHandleHolder, input: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>, training_space: &mut CuVectorDeref<f32>);
    fn backward_training(
        &self,
        cuda: &mut CudaHandleHolder,
        learning_rate: f32,
        layer_input: &CuVectorDeref<f32>,
        layer_output: &CuVectorDeref<f32>,
        front_signal: &CuVectorDeref<f32>,
        training_space: &mut CuVectorDeref<f32>,
        params_change: &mut CuVectorDeref<f32>,
        back_signal: Option<&mut CuVectorDeref<f32>>
    );

    fn save_params(&self, file_name: &str) -> CubrainResult<()> {
        let mut buffer = vec![0.0; self.params_count()];
        self.params().clone_to_host(&mut buffer);
        buffer.as_slice().save(file_name)
    }
    fn load_params(&mut self, file_name: &str) -> CubrainResult<()> {
        Ok(self.params_mut().clone_from_host(&<[f32]>::load(file_name)?))
    }

    fn clone_from(&mut self, to_clone_params: &CuVectorDeref<f32>) {
        assert_eq!(self.params_count(), to_clone_params.len());
        self.params_mut().clone_from_device(to_clone_params);
    }
    fn clone_randomized_from(&mut self, cuda: &mut CudaHandleHolder, to_clone_params: &CuVectorDeref<f32>, magnitude: f32) {
        assert_eq!(self.params_count(), to_clone_params.len());
        cuda.curand.generate_uniform_range(self.params_mut(), -magnitude, magnitude, &DEFAULT_STREAM); //TODO Choose generation
        self.params_mut().add(to_clone_params, &DEFAULT_STREAM);
    }
    fn clone_weighted_from(&mut self, cuda: &mut CudaHandleHolder, to_clone_params: &[&CuVectorDeref<f32>], weights: &[f32]) {
        assert_eq!(to_clone_params.len(), weights.len());
        self.params_mut().init(0.0, &DEFAULT_STREAM);
        for i in 0..to_clone_params.len() {
            cuda.cublas.axpy(weights[i], to_clone_params[i], self.params_mut());
        }
    }

}