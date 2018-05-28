
use super::*;
use cumath::*;
use CudaHandleHolder;
use ForwardInference;


pub struct DenseLayer {
    matrix: CuMatrixPtr<f32>,
    bias: Option<CuVectorPtr<f32>>,
    activation: Option<Activation>
}

impl ForwardInference for DenseLayer {

    fn input_len(&self) -> usize { self.matrix.rows() }
    fn output_len(&self) -> usize {
        self.matrix.cols()
    }
    fn workspace_len(&self) -> usize { 0 }

    fn forward_inference(&self, cuda: &mut CudaHandleHolder, input: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>, _workspace: &mut CuVectorDeref<f32>) {
        assert_eq!(input.len(), self.matrix.rows());
        assert_eq!(output.len(), self.matrix.cols());
        unsafe { cuda.cublas.mult_row_m(input, &self.matrix.deref(), output) }
        if let Some(ref bias) = self.bias {
            unsafe { cuda.cublas.axpy(1.0, bias.deref(), output) }
        }
        if let Some(ref activation) = self.activation {
            activation.forward(cuda, output)
        }
    }

}

impl Layer for DenseLayer {

    fn clone_structure(&self, params: CuVectorPtr<f32>) -> Box<Layer> {
        assert_eq!(params.len(), self.params_count());
        if let Some(ref bias) = self.bias {
            let mut iter = unsafe { params.deref().slice_iter() };
            let result = Box::new(DenseLayer {
                matrix: iter.next_matrix(self.matrix.rows(), self.matrix.cols()).unwrap().as_wrapped_ptr(),
                bias: Some(iter.next(bias.len()).unwrap().as_wrapped_ptr()),
                activation: if let Some(ref activation) = self.activation { Some(activation.clone()) } else { None },
            });
            assert_eq!(0, iter.len());
            result
        } else {
            Box::new(DenseLayer {
                matrix: unsafe { params.deref().matrix_slice(0, self.matrix.rows(), self.matrix.cols()).as_wrapped_ptr() },
                bias: None,
                activation: if let Some(ref activation) = self.activation { Some(activation.clone()) } else { None },
            })
        }
    }

    fn params_count(&self) -> usize {
        if let Some(ref bias) = self.bias {
            self.matrix.rows() * self.matrix.cols() + bias.len()
        } else {
            self.matrix.rows() * self.matrix.cols()
        }
    }


    fn forward_training(&self, cuda: &mut CudaHandleHolder, input: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>) {
        assert_eq!(input.len(), self.matrix.rows());
        assert_eq!(output.len(), self.matrix.cols());
        unsafe { cuda.cublas.mult_row_m(input, &self.matrix.deref(), output) }
        if let Some(ref bias) = self.bias {
            unsafe { cuda.cublas.axpy(1.0, &bias.deref(), output) }
        }
        if let Some(ref activation) = self.activation {
            activation.forward(cuda, output)
        }
    }
    fn backward_training(&self, cuda: &mut CudaHandleHolder, learning_rate: f32, momentum: f32,
                     layer_input: &CuVectorDeref<f32>, _layer_output: &CuVectorDeref<f32>, front_signal: &mut CuVectorDeref<f32>,
                     params_change: &mut CuVectorDeref<f32>, back_signal: Option<&mut CuVectorDeref<f32>>) {
        assert_eq!(layer_input.len(), self.input_len());
        assert_eq!(front_signal.len(), self.output_len());
        assert_eq!(params_change.len(), self.params_count());

        if let Some(ref activation) = self.activation {
            activation.backward(cuda, _layer_output, front_signal);
        }

        if let Some(ref bias) = self.bias {
            let mut iter = params_change.slice_mut_iter();
            let mut matrix_change = iter.next_matrix(self.matrix.rows(), self.matrix.cols()).unwrap();
            let mut bias_change = iter.next(bias.len()).unwrap();
            cuda.cublas.mult_col_row_rescaled(layer_input, front_signal, &mut matrix_change, -learning_rate, momentum);
            cuda.cublas.scal(&mut bias_change, momentum);
            cuda.cublas.axpy(-learning_rate, front_signal, &mut bias_change);

        } else {
            let mut matrix_change = params_change.matrix_slice_mut(0, self.input_len(), self.output_len());
            cuda.cublas.mult_col_row_rescaled(layer_input, front_signal, &mut matrix_change, -learning_rate, momentum);
        }

        if let Some(back_signal) = back_signal {
            assert!(back_signal.len() <= self.input_len());
            unsafe { cuda.cublas.mult_m_col(&self.matrix.deref().slice(0, 0, back_signal.len(), self.output_len()),
                                            front_signal, back_signal) }
        }
    }
}


#[derive(Clone, Debug, PartialEq)]
pub struct DenseLayerDescriptor {
    input_len: usize,
    output_len: usize,
    bias: bool,
    activation: Option<ActivationDescriptor>,
}
impl DenseLayerDescriptor {

    pub fn new(input_len: usize, output_len: usize, bias: bool, activation: Option<ActivationDescriptor>) -> DenseLayerDescriptor {
        DenseLayerDescriptor { input_len, output_len, bias, activation }
    }
    pub fn input_len(&self) -> usize { self.input_len }
    pub fn output_len(&self) -> usize { self.output_len }
    pub fn bias(&self) -> bool { self.bias }
    pub fn activation(&self) -> &Option<ActivationDescriptor> { &self.activation }

    pub fn create_layer(&self, params: CuVectorPtr<f32>) -> Box<Layer> {
        assert_eq!(self.params_count(), params.len());
        if self.bias {
            let mut iter = unsafe { params.deref().slice_iter() };
            let result = Box::new(DenseLayer {
                matrix: iter.next_matrix(self.input_len, self.output_len).unwrap().as_wrapped_ptr(),
                bias: Some(iter.next(self.output_len).unwrap().as_wrapped_ptr()),
                activation: if let Some(ref activation) = self.activation { Some(activation.build(self.output_len)) } else { None },
            });
            assert_eq!(0, iter.len());
            result
        } else {
            Box::new(DenseLayer {
                matrix: unsafe { params.deref().matrix_slice(0, self.input_len, self.output_len).as_wrapped_ptr() },
                bias: None,
                activation: if let Some(ref activation) = self.activation { Some(activation.build(self.output_len)) } else { None },
            })
        }
    }
    pub fn params_count(&self) -> usize {
        if self.bias {
            (1 + self.input_len) * self.output_len
        } else {
            self.input_len * self.output_len
        }
    }
}


#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    //#[ignore]
    fn test_one_layer() {

        let mut weights_holder = CuVector::<f32>::zero(4);
        let mut weights_ptr = weights_holder.as_wrapped_mut_ptr();

        let mut curand = CurandGenerator::new(CurandRngType::PseudoDefault).unwrap();
        curand.generate_uniform_range(&mut weights_holder, -1.0, 1.0, &DEFAULT_STREAM);

        let layer = Box::new(
            LayerDescriptor::dense(3, 1, true, Some(ActivationDescriptor::sigmoid()))
        ).create_layer(weights_holder.as_wrapped_ptr());


        let inputs = [
            CuVector::<f32>::from_host_data(&[0.0, 0.0, 1.0]),
            CuVector::<f32>::from_host_data(&[0.0, 1.0, 1.0]),
            CuVector::<f32>::from_host_data(&[1.0, 0.0, 1.0]),
            CuVector::<f32>::from_host_data(&[1.0, 1.0, 1.0]),
        ];
        let ideals = [
            CuVector::<f32>::from_host_data(&[0.0]),
            CuVector::<f32>::from_host_data(&[0.5]),
            CuVector::<f32>::from_host_data(&[0.5]),
            CuVector::<f32>::from_host_data(&[1.0]),
        ];

        let mut cuda = CudaHandleHolder::new(CurandRngType::PseudoDefault);
        let mut workspace = CuVector::<f32>::zero(layer.workspace_len());
        let mut output_buffer = CuVector::<f32>::zero(1);

        let mut output_signal_buffer = CuVector::<f32>::zero(1);
        let mut weights_change_buffer = CuVector::<f32>::zero(4);

        for iter in 0..1000 {
            let mut error = 0.0;
            for i in 0..4 {
                layer.forward_training(&mut cuda, &inputs[i], &mut output_buffer);

                CuVectorMath::<f32>::sub(&output_buffer, &ideals[i], &mut output_signal_buffer, &DEFAULT_STREAM);
                error += cuda.cublas.asum(&output_signal_buffer);
                layer.backward_training(&mut cuda, 0.1, 1.0, &inputs[i], &output_buffer,
                                     &mut output_signal_buffer,&mut weights_change_buffer, None);

            }
            unsafe { weights_ptr.deref_mut().add(&weights_change_buffer, &DEFAULT_STREAM); }
            weights_change_buffer.scl(0.9, &DEFAULT_STREAM);
            println!("Iteration {}, Error = {:.50}", iter, error);
        }


    }

    #[test]
    //#[ignore]
    fn test_xor() {
        let hidden_dimension = 3;

        let mut weights1_holder = CuVector::<f32>::zero(3*hidden_dimension);
        let mut weights2_holder = CuVector::<f32>::zero(hidden_dimension+1);
        let mut weights1_ptr = weights1_holder.as_wrapped_mut_ptr();
        let mut weights2_ptr = weights2_holder.as_wrapped_mut_ptr();

        {
            let mut curand = CurandGenerator::new(CurandRngType::PseudoDefault).unwrap();
            curand.generate_uniform_range(&mut weights1_holder, -1.0, 1.0, &DEFAULT_STREAM);
            curand.generate_uniform_range(&mut weights2_holder, -1.0, 1.0, &DEFAULT_STREAM);
        }

        let layer1 = Box::new(
            LayerDescriptor::dense(3, hidden_dimension, false, Some(ActivationDescriptor::sigmoid()))
        ).create_layer(weights1_holder.as_wrapped_ptr());
        let layer2 = Box::new(
            LayerDescriptor::dense(hidden_dimension+1, 1, false, Some(ActivationDescriptor::sigmoid()))
        ).create_layer(weights2_holder.as_wrapped_ptr());


        let inputs = [
            CuVector::<f32>::from_host_data(&[0.0, 0.0, 1.0]),
            CuVector::<f32>::from_host_data(&[0.0, 1.0, 1.0]),
            CuVector::<f32>::from_host_data(&[1.0, 0.0, 1.0]),
            CuVector::<f32>::from_host_data(&[1.0, 1.0, 1.0]),
        ];
        let ideals = [
            CuVector::<f32>::from_host_data(&[0.0]),
            CuVector::<f32>::from_host_data(&[1.0]),
            CuVector::<f32>::from_host_data(&[1.0]),
            CuVector::<f32>::from_host_data(&[0.0]),
        ];

        let mut cuda = CudaHandleHolder::new(CurandRngType::PseudoDefault);
        let mut workspace = CuVector::<f32>::zero(1);

        let mut hidden_buffer = CuVector::<f32>::new(1.0, hidden_dimension+1);
        let mut output_buffer = CuVector::<f32>::zero(1);

        let mut output_signal_buffer = CuVector::<f32>::zero(1);
        let mut hidden_signal_buffer = CuVector::<f32>::zero(hidden_dimension);

        let mut weights_change1_buffer = CuVector::<f32>::zero(3*hidden_dimension);
        let mut weights_change2_buffer = CuVector::<f32>::zero((hidden_dimension+1)*1);

        for iter in 0..10000 {
            let mut error = 0.0;
            for i in 0..1 {
                layer1.forward_training(&mut cuda, &inputs[i], &mut hidden_buffer.slice_mut(0, hidden_dimension));
                layer2.forward_training(&mut cuda, &hidden_buffer, &mut output_buffer);

                CuVectorMath::<f32>::sub(&output_buffer, &ideals[i], &mut output_signal_buffer, &DEFAULT_STREAM);
                error += cuda.cublas.asum(&output_signal_buffer);

                layer2.backward_training(&mut cuda, 0.1, 1.0, &hidden_buffer, &output_buffer,
                                     &mut output_signal_buffer, &mut weights_change2_buffer, Some(&mut hidden_signal_buffer));

                layer1.backward_training(&mut cuda, 0.1, 1.0, &inputs[i], &hidden_buffer.slice_mut(0, hidden_dimension),
                                     &mut hidden_signal_buffer,&mut weights_change1_buffer, None);

            }
            println!("Iteration {}, Error = {:.20}", iter, error);

            unsafe {
                weights1_ptr.deref_mut().add(&weights_change1_buffer, &DEFAULT_STREAM);
                weights2_ptr.deref_mut().add(&weights_change2_buffer, &DEFAULT_STREAM);
            }

            weights_change1_buffer.scl(0.9, &DEFAULT_STREAM);
            weights_change2_buffer.scl(0.9, &DEFAULT_STREAM);

        }

    }

}