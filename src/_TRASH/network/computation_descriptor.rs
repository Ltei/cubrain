
use std::convert::Into;
use super::{Computation, ComputationParams, DenseLayerDescriptor, SigmoidDescriptor, DenseLayer, SigmoidLayer};
use cumath::*;
use cumath_nn::*;



#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ComputationDescriptor {
    input_len: usize,
    output_len: usize,
    params_count: usize,
    workspace_len: usize,
    training_space_len: usize,
    backpropagation_need_input: bool,
    backpropagation_need_output: bool,
    info: ComputationInfo,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum ComputationInfo {
    DenseLayer,
    Sigmoid,
}

impl ComputationDescriptor {

    pub fn input_len(&self) -> usize {
        self.input_len
    }
    pub fn output_len(&self) -> usize {
        self.output_len
    }
    pub fn params_count(&self) -> usize { self.params_count }
    pub fn workspace_len(&self) -> usize { self.workspace_len }
    pub fn training_space_len(&self) -> usize { self.training_space_len }

    pub fn backpropagation_need_input(&self) -> bool {
        self.backpropagation_need_input
    }
    pub fn backpropagation_need_output(&self) -> bool {
        self.backpropagation_need_output
    }

    pub fn info(&self) -> ComputationInfo {
        self.info
    }

    pub fn into_computation_owning(self) -> Box<Computation> {
        Box::new(match self.info() {
            ComputationInfo::DenseLayer => {
                let params = CuVector::<f32>::zero(self.input_len() * self.output_len());
                DenseLayer {
                    params: ComputationParams::Owned(params),
                    matrix: params.matrix_slice(0, self.input_len(), self.output_len()).as_wrapped_ptr(),
                    descriptor: self,
                }
            },
            ComputationInfo::Sigmoid => {
                SigmoidLayer {
                    params: CuVector::<f32>::zero(0),
                    function: CuActivationDescriptor::new(CudnnActivationMode::Sigmoid, 0.0),
                    io_shape: CuTensorDescriptor::fully_packed(&[1, 1, 1, self.input_len() as i32]),
                    descriptor: self,
                }
            },
        })
    }

    pub fn into_computation_borrowing(self, mut params: CuVectorMutPtr<f32>) -> Box<Computation> {
        assert_eq!(params.len(), self.params_count());
        Box::new(match self.info() {
            ComputationInfo::DenseLayer => {
                DenseLayer {
                    matrix: unsafe { params.deref_mut() }.matrix_slice(0, self.input_len(), self.output_len()).as_wrapped_ptr(),
                    params: ComputationParams::Borrowed(params),
                    descriptor: self,
                }
            },
            ComputationInfo::Sigmoid => {
                SigmoidLayer {
                    params: CuVector::<f32>::zero(0),
                    function: CuActivationDescriptor::new(CudnnActivationMode::Sigmoid, 0.0),
                    io_shape: CuTensorDescriptor::fully_packed(&[1, 1, 1, self.input_len() as i32]),
                    descriptor: self,
                }
            },
        })
    }

}