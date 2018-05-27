

use super::*;
use cumath::*;
use cumath_nn::*;
use CudaHandleHolder;
use super::{Computation, ComputationParams};
use meta::CubrainResult;




pub struct SigmoidLayer {
    descriptor: ComputationDescriptor,
    params: CuVector<f32>, // Phantom data since len will always be 0
    function: CuActivationDescriptor,
    io_shape: CuTensorDescriptor<f32>,
}

impl Computation for SigmoidLayer {

    fn descriptor(&self) -> &ComputationDescriptor { &self.descriptor }
    fn params(&self) -> &CuVectorDeref<f32> { &self.params }
    fn params_mut(&mut self) -> &mut CuVectorDeref<f32> { &mut self.params }

    fn forward_inference(&self, cuda: &mut CudaHandleHolder, input: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>, workspace: &mut CuVectorDeref<f32>) {
        assert_eq!(input.len(), self.gate_len);
        assert_eq!(output.len(), self.gate_len);
        self.function.forward(&mut cuda.cudnn, &self.io_shape.link(input), 1.0, &mut self.io_shape.link_mut(output), 0.0);
    }

    fn forward_training(&self, cuda: &mut CudaHandleHolder, input: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>, training_space: &mut CuVectorDeref<f32>) {
        assert_eq!(input.len(), self.gate_len);
        assert_eq!(output.len(), self.gate_len);
        self.function.forward(&mut cuda.cudnn, &self.io_shape.link(input), 1.0, &mut self.io_shape.link_mut(output), 0.0);
    }

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
    ) {
        assert_eq!(0, params_change.len());

        if let Some(back_signal) = back_signal {
            self.function.backward(&mut cuda.cudnn,
                                   1.0, 0.0,
                                   &self.io_shape.link(layer_output),
                                   &self.io_shape.link(front_signal),
                                   &self.io_shape.link(layer_output),
                                   &mut self.io_shape.link_mut(back_signal));
        }
    }

}



#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SigmoidDescriptor {
    gate_len: usize,
}
impl SigmoidDescriptor {
    pub fn gate_len(&self) -> usize { self.gate_len }
}