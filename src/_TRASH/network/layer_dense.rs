

use cumath::*;
use super::{Computation, ComputationParams, ComputationDescriptor};
use CudaHandleHolder;
use meta::CubrainResult;




pub struct DenseLayer {
    descriptor: ComputationDescriptor,
    params: ComputationParams,
    matrix: CuMatrixPtr<f32>,
}

impl Computation for DenseLayer {

    fn descriptor(&self) -> &ComputationDescriptor {
        &self.descriptor
    }

    fn params(&self) -> &CuVectorDeref<f32> {
        match self.params {
            ComputationParams::Owned(ref params) => params,
            ComputationParams::Borrowed(ref params) => unsafe { params.deref() },
        }
    }

    fn params_mut(&mut self) -> &mut CuVectorDeref<f32> {
        match self.params {
            ComputationParams::Owned(ref mut params) => params,
            ComputationParams::Borrowed(ref mut params) => unsafe { params.deref_mut() },
        }
    }

    fn forward_inference(&self, cuda: &mut CudaHandleHolder, input: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>, workspace: &mut CuVectorDeref<f32>) {
        assert_eq!(input.len(), self.input_len());
        assert_eq!(output.len(), self.output_len());

        cuda.cublas.mult_row_m(input, unsafe { self.matrix.deref() }, output);
    }

    fn forward_training(&self, cuda: &mut CudaHandleHolder, input: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>, training_space: &mut CuVectorDeref<f32>) {
        assert_eq!(input.len(), self.input_len());
        assert_eq!(output.len(), self.output_len());

        cuda.cublas.mult_row_m(input, unsafe { self.matrix.deref() }, output);
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
        assert_eq!(layer_input.len(), self.input_len());
        assert_eq!(front_signal.len(), self.output_len());
        assert_eq!(params_change.len(), self.params_count());

        let mut weights_change = params_change.matrix_slice_mut(0, self.input_len(), self.output_len());
        cuda.cublas.mult_col_row_rescaled(layer_input, front_signal, &mut weights_change, -learning_rate, 1.0);

        if let Some(back_signal) = back_signal {
            assert!(back_signal.len() <= self.input_len());
            unsafe { cuda.cublas.mult_m_col(&self.matrix.deref().slice(0, 0, back_signal.len(), self.output_len()),
                                            front_signal, back_signal) }
        }
    }

}


#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DenseLayerDescriptor {
    input_len: usize,
    output_len: usize,
}
impl DenseLayerDescriptor {
    pub fn input_len(&self) -> usize { self.input_len }
    pub fn output_len(&self) -> usize { self.output_len }
}