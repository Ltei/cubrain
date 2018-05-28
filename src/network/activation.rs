
use cumath::*;
use cumath_nn::*;
use CudaHandleHolder;




pub struct Activation {
    function: CuActivationDescriptor,
    io_shape: CuTensorDescriptor<f32>,
}

impl Activation {

    pub fn forward(&self, cuda: &mut CudaHandleHolder, input: &mut CuVectorDeref<f32>) {
        self.function.forward_inplace(&mut cuda.cudnn, &mut self.io_shape.link_mut(input), 1.0, 0.0);
    }
    pub fn backward(&self, cuda: &mut CudaHandleHolder, output: &CuVectorDeref<f32>, signal: &mut CuVectorDeref<f32>) {
        self.function.backward_inplace(&mut cuda.cudnn, 1.0, 0.0,
                               &self.io_shape.link(output),
                               &mut self.io_shape.link_mut(signal));
    }

    pub fn clone(&self) -> Activation {
        let info = self.function.get_info();
        Activation {
            function: CuActivationDescriptor::new(info.mode, info.coef),
            io_shape: self.io_shape.clone(),
        }
    }

}



#[derive(Clone, Debug, PartialEq)]
pub struct ActivationDescriptor {
    mode: CudnnActivationMode,
    coef: f64,
}
impl ActivationDescriptor {

    pub fn sigmoid() -> ActivationDescriptor {
        ActivationDescriptor {
            mode: CudnnActivationMode::Sigmoid,
            coef: 0.0,
        }
    }
    pub fn tanh() -> ActivationDescriptor {
        ActivationDescriptor {
            mode: CudnnActivationMode::Tanh,
            coef: 0.0,
        }
    }
    pub fn relu() -> ActivationDescriptor {
        ActivationDescriptor {
            mode: CudnnActivationMode::Relu,
            coef: 0.0,
        }
    }

    pub fn build(&self, len: usize) -> Activation {
        Activation {
            function: CuActivationDescriptor::new(self.mode, self.coef),
            io_shape: CuTensorDescriptor::fully_packed(&[1, 1, 1, len as i32]),
        }
    }

}