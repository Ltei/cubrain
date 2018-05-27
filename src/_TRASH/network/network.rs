
use super::*;
use cumath::*;
use CudaHandleHolder;
use std::mem;




pub struct Network {
    params: ComputationParams,
    layers: Vec<Box<Computation>>,
    input_len: usize,
    output_len: usize,
    biggest_layer_workspace_len: usize,
    biggest_hidden_len: usize,
}

impl Computation for Network {

    fn input_len(&self) -> usize {
        self.input_len
    }

    fn output_len(&self) -> usize {
        self.output_len
    }

    fn params_count(&self) -> usize {
        self.layers.iter().fold(0, |acc, x| acc + x.params_count())
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

    fn workspace_len(&self) -> usize {
        if self.layers.len() == 1 {
            self.biggest_layer_workspace_len
        } else if self.layers.len() == 2 {
            self.biggest_layer_workspace_len + self.biggest_hidden_len
        } else {
            self.biggest_layer_workspace_len + 2 * self.biggest_hidden_len
        }
    }

    fn training_space_len(&self) -> usize {
        0
    }

    fn clone_structure_owning(&self) -> Box<Computation> {
        let params = CuVector::<f32>::zero(self.params_count());
        let layers = {
            let mut iter = unsafe { params.slice_iter() };
            let result = self.layers.iter().map(|x| {
                x.clone_structure_borrowing(iter.next(x.params_count()).unwrap().as_wrapped_mut_ptr())
            }).collect();
            assert_eq!(iter.len(), 0);
            result
        };
        Box::new(Network {
            params: ComputationParams::Owned(params),
            input_len: self.input_len,
            output_len: self.output_len,
            biggest_layer_workspace_len: self.layers.iter().fold(0, |acc, x| if x.workspace_len() > acc { x.workspace_len() } else { acc }),
            biggest_hidden_len: self.layers.iter().skip(1).fold(0, |acc, x| if x.input_len() > acc { x.input_len() } else { acc }),
            layers,
        })
    }

    fn clone_structure_borrowing(&self, params: CuVectorMutPtr<f32>) -> Box<Computation> {
        let layers = {
            let mut iter = unsafe { params.deref().slice_iter() };
            let result = self.layers.iter().map(|x| {
                x.clone_structure_borrowing(iter.next(x.params_count()).unwrap().as_wrapped_mut_ptr())
            }).collect();
            assert_eq!(iter.len(), 0);
            result
        };
        Box::new(Network {
            params: ComputationParams::Borrowed(params),
            input_len: self.input_len,
            output_len: self.output_len,
            biggest_layer_workspace_len: self.layers.iter().fold(0, |acc, x| if x.workspace_len() > acc { x.workspace_len() } else { acc }),
            biggest_hidden_len: self.layers.iter().skip(1).fold(0, |acc, x| if x.input_len() > acc { x.input_len() } else { acc }),
            layers,
        })
    }

    fn forward_inference(&self, cuda: &mut CudaHandleHolder, input: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>, workspace: &mut CuVectorDeref<f32>) {
        assert_eq!(input.len(), self.input_len());
        assert_eq!(output.len(), self.output_len());
        assert!(workspace.len() >= self.workspace_len());

        let mut workspace_iter = workspace.slice_mut_iter();
        let mut layer_workspace = workspace_iter.next(self.biggest_layer_workspace_len).unwrap();

        if self.layers.len() == 1 {
            self.layers[0].forward_inference(cuda, input, output, &mut layer_workspace);

        } else if self.layers.len() == 2 {
            let mut tmp_hidden = workspace_iter.next(self.biggest_hidden_len).unwrap();
            self.layers[0].forward_inference(cuda, input, &mut tmp_hidden, &mut layer_workspace);
            self.layers[1].forward_inference(cuda, &tmp_hidden, output, &mut layer_workspace);

        } else {
            let last_idx = self.layers.len() -1;
            let mut tmp_hidden1 = workspace_iter.next(self.biggest_hidden_len).unwrap();
            let mut tmp_hidden2 = workspace_iter.next(self.biggest_hidden_len).unwrap();

            self.layers[0].forward_inference(cuda,
                                             input,
                                             &mut tmp_hidden1.slice_mut(0, self.layers[0].output_len()),
                                             &mut layer_workspace);
            for i in 1..last_idx {
                self.layers[i].forward_inference(cuda,
                                                 &tmp_hidden1.slice(0, self.layers[i].input_len()),
                                                 &mut tmp_hidden2.slice_mut(0, self.layers[i].output_len()),
                                                 &mut layer_workspace);
                mem::swap(&mut tmp_hidden1, &mut tmp_hidden2);
            }
            self.layers[last_idx].forward_inference(cuda,
                                                    &tmp_hidden1.slice(0, self.layers[last_idx].input_len()),
                                                    output,
                                                    &mut layer_workspace);

        }
    }

    fn forward_training(&self, cuda: &mut CudaHandleHolder, input: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>, training_space: &mut CuVectorDeref<f32>) {
        unimplemented!()
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
        unimplemented!()
    }

}


#[derive(Clone, PartialEq, Eq, Debug)]
pub struct NetworkDescriptor {
    pub(super) layers: Vec<ComputationDescriptor>,
}
impl NetworkDescriptor {

    pub fn new(layers: Vec<ComputationDescriptor>) -> NetworkDescriptor {
        assert!(layers.len() > 0);
        {
            let mut iter = layers.iter();
            let mut last_output = iter.next().unwrap().output_len();
            iter.for_each(|x| {
                assert_eq!(last_output, x.input_len());
                last_output = x.output_len();
            });
        }
        NetworkDescriptor { layers }
    }

    pub fn input_len(&self) -> usize {
        self.layers[0].input_len()
    }
    pub fn output_len(&self) -> usize {
        self.layers.iter().last().unwrap().output_len()
    }
    pub fn weights_count(&self) -> usize {
        self.layers.iter().fold(0, |acc, x| { acc + x.params_count() })
    }
    pub fn create_network(&self, data: CuVectorPtr<f32>) -> Network {
        assert_eq!(data.len(), self.weights_count());
        let mut iter = unsafe { data.deref().slice_iter() };
        let output = Network {
            layers: self.layers.iter().map(|x| {
                x.create_layer(iter.next(x.params_count()).unwrap().as_wrapped_ptr())
            }).collect(),
            input_len: self.layers.first().unwrap().input_len(),
            output_len: self.layers.last().unwrap().output_len(),
        };
        assert_eq!(iter.len(), 0);
        output
    }
    pub fn duplicate(&self) -> NetworkDescriptor {
        NetworkDescriptor { layers: self.layers.iter().map(|x| x.duplicate()).collect() }
    }
}


#[cfg(test)]
mod tests {

    use cumath::*;
    use cumath_nn::*;
    use gate::*;

    #[test]
    fn test_xor2() {
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

        let network = NetworkBuilder::new_dense(3)
            //.activation(CudnnActivationMode::Tanh, 1.0)
            //.convolution(2, 2, vec![ConvolutionKernelDescriptor::new(1, 1, 1, 1), ConvolutionKernelDescriptor::new(1, 1, 1, 1), ConvolutionKernelDescriptor::new(1, 1, 1, 1), ConvolutionKernelDescriptor::new(1, 1, 1, 1)])
            .activation(CudnnActivationMode::Sigmoid, 1.0)
            .dense(4)
            .activation(CudnnActivationMode::Sigmoid, 1.0)
            .build(1);

        println!("Network descriptor :\n{:?}", network);

        let mut cuda = CudaHandleHolder::new(CurandRngType::PseudoDefault);
        let mut weights_holder = CuVector::<f32>::zero(network.weights_count());
        let network = network.create_network(weights_holder.as_wrapped_ptr());
        cuda.curand.generate_uniform_range(&mut weights_holder, -1.0, 1.0, &DEFAULT_STREAM);

        let mut workspace = network.create_workspace();
        let mut training_space = network.create_training_space();
        let mut output_buffer = CuVector::<f32>::zero(1);
        let mut output_signal_buffer = CuVector::<f32>::zero(1);
        let mut weights_change_buffer = CuVector::<f32>::zero(network.weights_count());

        for iter in 0..5000 {
            let mut error = 0.0;
            for i in 0..4 {
                network.forward_training(&mut cuda, &mut workspace, &inputs[i], &mut output_buffer);

                CuVectorMath::<f32>::sub(&output_buffer, &ideals[i], &mut output_signal_buffer, &DEFAULT_STREAM);
                error += cuda.cublas.asum(&output_signal_buffer);

                network.backpropagate(&mut cuda, 0.1, 1.0,
                                      &mut workspace, &inputs[i],
                                      &mut output_buffer, &output_signal_buffer,
                                      &mut training_space, &mut weights_change_buffer, None);

            }
            println!("Iteration {}, Error = {:.20}", iter, error);

            weights_holder.add(&weights_change_buffer, &DEFAULT_STREAM);

            weights_change_buffer.scl(0.9, &DEFAULT_STREAM);

        }
    }

    #[test]
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
            for i in 0..4 {
                layer1.compute(&mut cuda, &mut workspace, &inputs[i], &mut hidden_buffer.slice_mut(0, hidden_dimension));
                layer2.compute(&mut cuda, &mut workspace, &hidden_buffer, &mut output_buffer);

                CuVectorMath::<f32>::sub(&output_buffer, &ideals[i], &mut output_signal_buffer, &DEFAULT_STREAM);
                error += cuda.cublas.asum(&output_signal_buffer);

                layer2.backpropagate(&mut cuda, &mut workspace, 0.1, 1.0, &hidden_buffer, &mut output_buffer,
                                     &output_signal_buffer, &mut weights_change2_buffer, Some(&mut hidden_signal_buffer));

                layer1.backpropagate(&mut cuda, &mut workspace, 0.1, 1.0, &inputs[i], &mut hidden_buffer.slice_mut(0, hidden_dimension),
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