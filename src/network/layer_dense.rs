
use super::*;
use cumath::*;
use CudaHandleHolder;
use ForwardInference;


pub struct DenseLayer {
    pub(super) weights: CuMatrixPtr<f32>,
}

impl ForwardInference for DenseLayer {

    fn input_len(&self) -> usize { self.weights.rows() }
    fn output_len(&self) -> usize {
        self.weights.cols()
    }
    fn workspace_len(&self) -> usize { 0 }

    fn forward_inference(&self, cuda: &mut CudaHandleHolder, input: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>, _workspace: &mut CuVectorDeref<f32>) {
        assert_eq!(input.len(), self.weights.rows());
        assert_eq!(output.len(), self.weights.cols());
        unsafe { cuda.cublas.mult_row_m(input, &self.weights.deref(), output) }
    }

}

impl Layer for DenseLayer {

    fn clone_structure(&self, data: CuVectorPtr<f32>) -> Box<Layer> {
        Box::new(DenseLayer {
            weights: unsafe { data.deref().matrix_slice(0, self.weights.rows(), self.weights.cols()).as_wrapped_ptr() },
        })
    }

    fn params_count(&self) -> usize {
        self.weights.rows() * self.weights.cols()
    }


    fn forward_training(&self, cuda: &mut CudaHandleHolder, input: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>) {
        assert_eq!(input.len(), self.weights.rows());
        assert_eq!(output.len(), self.weights.cols());
        unsafe { cuda.cublas.mult_row_m(input, &self.weights.deref(), output) }
    }
    fn backward_training(&self, cuda: &mut CudaHandleHolder, learning_rate: f32, momentum: f32,
                     layer_input: &CuVectorDeref<f32>, _layer_output: &mut CuVectorDeref<f32>, front_signal: &CuVectorDeref<f32>,
                     weights_change: &mut CuVectorDeref<f32>, back_signal: Option<&mut CuVectorDeref<f32>>) {
        assert_eq!(layer_input.len(), self.input_len());
        assert_eq!(front_signal.len(), self.output_len());
        assert_eq!(weights_change.len(), self.params_count());

        let mut weights_change = weights_change.matrix_slice_mut(0, self.input_len(), self.output_len());
        cuda.cublas.mult_col_row_rescaled(layer_input, front_signal, &mut weights_change, -learning_rate, momentum);

        if let Some(back_signal) = back_signal {
            assert!(back_signal.len() <= self.input_len());
            unsafe { cuda.cublas.mult_m_col(&self.weights.deref().slice(0, 0, back_signal.len(), self.output_len()),
                                                 front_signal, back_signal) }
        }
    }
}


#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DenseLayerDescriptor {
    pub(super) input_len: usize,
    pub(super) output_len: usize,
}
impl DenseLayerDescriptor {
    pub fn create_layer(&self, data: CuVectorPtr<f32>) -> Box<Layer> {
        Box::new(DenseLayer {
            weights: unsafe { data.deref().matrix_slice(0, self.input_len, self.output_len).as_wrapped_ptr() },
        })
    }
}


#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    #[ignore]
    fn test_one_layer() {

        let mut weights_holder = CuVector::<f32>::zero(3);
        let mut weights_ptr = weights_holder.as_wrapped_mut_ptr();

        let mut curand = CurandGenerator::new(CurandRngType::PseudoDefault).unwrap();
        curand.generate_uniform_range(&mut weights_holder, -1.0, 1.0, &DEFAULT_STREAM);

        let layer = Box::new(
            LayerDescriptor::dense(3, 1)
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
        let mut weights_change_buffer = CuVector::<f32>::zero(3);

        for iter in 0..1000 {
            let mut error = 0.0;
            for i in 0..4 {
                layer.forward_training(&mut cuda, &inputs[i], &mut output_buffer);

                CuVectorMath::<f32>::sub(&output_buffer, &ideals[i], &mut output_signal_buffer, &DEFAULT_STREAM);
                error += cuda.cublas.asum(&output_signal_buffer);
                layer.backward_training(&mut cuda, 0.1, 1.0, &inputs[i], &mut output_buffer,
                                     &output_signal_buffer,&mut weights_change_buffer, None);

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
            LayerDescriptor::dense(3, hidden_dimension)
        ).create_layer(weights1_holder.as_wrapped_ptr());
        let layer2 = Box::new(
            LayerDescriptor::dense(hidden_dimension+1, 1)
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

                layer2.backward_training(&mut cuda, 0.1, 1.0, &hidden_buffer, &mut output_buffer,
                                     &output_signal_buffer, &mut weights_change2_buffer, Some(&mut hidden_signal_buffer));

                layer1.backward_training(&mut cuda, 0.1, 1.0, &inputs[i], &mut hidden_buffer.slice_mut(0, hidden_dimension),
                                     &hidden_signal_buffer,&mut weights_change1_buffer, None);

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