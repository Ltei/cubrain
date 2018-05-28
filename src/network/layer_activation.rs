
use super::*;
use cumath::*;
use cumath_nn::*;
use CudaHandleHolder;
use ForwardInference;




pub struct ActivationLayer {
    pub(super) function: CuActivationDescriptor,
    pub(super) gate_len: usize,
    pub(super) io_shape: CuTensorDescriptor<f32>,
}

impl ForwardInference for ActivationLayer {

    fn input_len(&self) -> usize {
        self.gate_len
    }
    fn output_len(&self) -> usize {
        self.gate_len
    }
    fn workspace_len(&self) -> usize { 0 }

    fn forward_inference(&self, cuda: &mut CudaHandleHolder, input: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>, _workspace: &mut CuVectorDeref<f32>) {
        assert_eq!(input.len(), self.gate_len);
        assert_eq!(output.len(), self.gate_len);
        self.function.forward(&mut cuda.cudnn, &self.io_shape.link(input), 1.0, &mut self.io_shape.link_mut(output), 0.0);
    }

}

impl Layer for ActivationLayer {

    fn clone_structure(&self, _weights: CuVectorPtr<f32>) -> Box<Layer> {
        let info = self.function.get_info();
        Box::new(ActivationLayer {
            function: CuActivationDescriptor::new(info.mode, info.coef),
            gate_len: self.gate_len,
            io_shape: self.io_shape.clone(),
        })
    }


    fn params_count(&self) -> usize { 0 }

    fn forward_training(&self, cuda: &mut CudaHandleHolder, input: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>) {
        assert_eq!(input.len(), self.gate_len);
        assert_eq!(output.len(), self.gate_len);
        self.function.forward(&mut cuda.cudnn, &self.io_shape.link(input), 1.0, &mut self.io_shape.link_mut(output), 0.0);
    }
    fn backward_training(&self, cuda: &mut CudaHandleHolder, _learning_rate: f32, _momentum: f32,
                         _layer_input: &CuVectorDeref<f32>, layer_output: &CuVectorDeref<f32>, front_signal: &mut CuVectorDeref<f32>,
                         params_change: &mut CuVectorDeref<f32>, back_signal: Option<&mut CuVectorDeref<f32>>) {
        assert_eq!(0, params_change.len());

        if let Some(back_signal) = back_signal {
            self.function.backward(&mut cuda.cudnn, 1.0, 0.0,
                                   &self.io_shape.link(layer_output),
                                   &self.io_shape.link(front_signal),
                                   &mut self.io_shape.link_mut(back_signal));
        }
    }

}


#[derive(Clone, Debug)]
pub struct ActivationLayerDescriptor {
    pub(super) mode: CudnnActivationMode,
    pub(super) coef: f64,
    pub(super) len: usize,
}
impl ActivationLayerDescriptor {
    pub fn create_layer(&self, _data: CuVectorPtr<f32>) -> Box<Layer> {
        Box::new(ActivationLayer {
            function: CuActivationDescriptor::new(self.mode, self.coef),
            gate_len: self.len,
            io_shape: CuTensorDescriptor::fully_packed(&[1, 1, 1, self.len as i32]),
        })
    }
}