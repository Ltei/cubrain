
use std::mem;
use super::*;
use cumath::*;
use CudaHandleHolder;
use CloneStructure;
use GetParams;
use ForwardInference;




pub struct Network {
    params: CuVector<f32>,
    layers: Vec<Box<Layer>>,
    input_len: usize,
    output_len: usize,
    biggest_layer_workspace_len: usize,
    biggest_hidden_len: usize,
}

impl CloneStructure for Network {
    fn clone_structure(&self) -> Network {
        let params = CuVector::<f32>::zero(self.params().len());
        Network {
            layers: {
                let mut iter = params.slice_iter();
                let result = self.layers.iter().map(|x| {
                    x.clone_structure(iter.next(x.params_count()).unwrap().as_wrapped_ptr())
                }).collect();
                assert_eq!(iter.len(), 0);
                result
            },
            input_len: self.input_len,
            output_len: self.output_len,
            biggest_layer_workspace_len: self.biggest_layer_workspace_len,
            biggest_hidden_len: self.biggest_hidden_len,
            params,
        }
    }
}

impl GetParams for Network {
    fn params(&self) -> &CuVectorDeref<f32> { &self.params }
    fn params_mut(&mut self) -> &mut CuVectorDeref<f32> { &mut self.params }
}


impl ForwardInference for Network {

    fn input_len(&self) -> usize {
        self.input_len
    }
    fn output_len(&self) -> usize {
        self.output_len
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

}

impl Network {

    pub fn create_workspace(&self) -> NetworkWorkspace {
        NetworkWorkspace {
            hidden_outputs: if self.layers.len() > 1 {
                self.layers[1..].iter().map(|x| CuVector::<f32>::zero(x.input_len())).collect()
            } else {
                Vec::new()
            },
            layer_workspace: {
                let mut biggest_len = 0;
                for layer in &self.layers {
                    let len = layer.workspace_len();
                    if len > biggest_len { biggest_len = len }
                }
                CuVector::<f32>::zero(biggest_len)
            }
        }
    }
    pub fn create_training_space(&self) -> NetworkTrainingSpace {
        let workspace = self.create_workspace();
        NetworkTrainingSpace { hidden_signals: workspace.hidden_outputs }
    }

    pub fn forward_training(&self, cuda: &mut CudaHandleHolder, workspace: &mut NetworkWorkspace, input: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>) {
        if self.layers.len() > 1 {
            let mut output_iter = workspace.hidden_outputs.iter_mut();

            let mut last_output = output_iter.next().unwrap();
            self.layers[0].forward_training(cuda, input, &mut last_output.slice_mut(0, self.layers[0].output_len()));

            for layer in &self.layers[1..self.layers.len()-1] {
                let output = output_iter.next().unwrap();
                layer.forward_training(cuda, &last_output, &mut output.slice_mut(0, layer.output_len()));
                last_output = output;
            }

            assert_eq!(output_iter.len(), 0);

            self.layers[self.layers.len()-1].forward_training(cuda, &last_output, output);
        } else {
            self.layers[0].forward_training(cuda, input, output);
        }
    }

    pub fn backpropagate(&self, cuda: &mut CudaHandleHolder, learning_rate: f32, momentum: f32, workspace: &mut NetworkWorkspace,
                         layer_input: &CuVectorDeref<f32>,
                         layer_output: &CuVectorDeref<f32>,
                         front_signal: &mut CuVectorDeref<f32>,
                         training_space: &mut NetworkTrainingSpace,
                         weights_change: &mut CuVectorDeref<f32>,
                         back_signal: Option<&mut CuVectorDeref<f32>>) {
        if self.layers.len() > 1 {

            let last_layer_idx = self.layers.len()-1;
            let mut weights_change_iter = weights_change.slice_mut_iter();
            let mut hidden_output_iter = workspace.hidden_outputs.iter_mut().rev();
            let mut hidden_signals_iter = training_space.hidden_signals.iter_mut().rev();

            let gate = &self.layers[last_layer_idx];
            let mut last_hidden_output = hidden_output_iter.next().unwrap();
            let mut last_hidden_signal = hidden_signals_iter.next().unwrap();
            gate.backward_training(cuda, learning_rate, momentum,
                               &last_hidden_output,
                               layer_output, front_signal,
                               &mut weights_change_iter.last(gate.params_count()).unwrap(),
                               Some(&mut last_hidden_signal));

            for i in (1..last_layer_idx).rev() {
                let gate = &self.layers[i];
                let mut hidden_output = hidden_output_iter.next().unwrap();
                let mut hidden_signal = hidden_signals_iter.next().unwrap();
                gate.backward_training(cuda, learning_rate, momentum,
                                   &hidden_output,
                                   &last_hidden_output,
                                   &mut last_hidden_signal,
                                   &mut weights_change_iter.last(gate.params_count()).unwrap(),
                                   Some(&mut hidden_signal));
                last_hidden_output = hidden_output;
                last_hidden_signal = hidden_signal;
            }

            let gate = &self.layers[0];
            gate.backward_training(cuda, learning_rate, momentum,
                               layer_input,
                               &last_hidden_output,
                               &mut last_hidden_signal,
                               &mut weights_change_iter.last(gate.params_count()).unwrap(),
                               back_signal);

            assert_eq!(weights_change_iter.len(), 0);
            assert_eq!(hidden_output_iter.len(), 0);
            assert_eq!(hidden_signals_iter.len(), 0);

        } else {
            self.layers[0].backward_training(cuda, learning_rate, momentum, layer_input, layer_output, front_signal, weights_change, back_signal);
        }
    }

}




pub struct NetworkWorkspace {
    hidden_outputs: Vec<CuVector<f32>>,
    layer_workspace: CuVector<f32>,
}
pub struct NetworkTrainingSpace {
    hidden_signals: Vec<CuVector<f32>>,
}


#[derive(Clone, Debug)]
pub struct NetworkDescriptor {
    pub(super) layers: Vec<LayerDescriptor>,
}
impl NetworkDescriptor {

    pub fn new(layers: Vec<LayerDescriptor>) -> NetworkDescriptor {
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
    pub fn create_network(&self) -> Network {
        let params = CuVector::<f32>::zero(self.weights_count());
        let layers = {
            let mut iter = params.slice_iter();
            let result = self.layers.iter().map(|x| {
                x.create_layer(iter.next(x.params_count()).unwrap().as_wrapped_ptr())
            }).collect::<Vec<_>>();
            assert_eq!(iter.len(), 0);
            result
        };
        let output = Network {
            input_len: self.layers.first().unwrap().input_len(),
            output_len: self.layers.last().unwrap().output_len(),
            biggest_layer_workspace_len: layers.iter().fold(0, |acc, x| if x.workspace_len() > acc { x.workspace_len() } else { acc }),
            biggest_hidden_len: layers.iter().skip(1).fold(0, |acc, x| if x.input_len() > acc { x.input_len() } else { acc }),
            params,
            layers,

        };
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
    use NetworkBuilder;
    use ActivationDescriptor;
    use CudaHandleHolder;
    use LayerDescriptor;
    use GetParams;

    #[test]
    fn test_xor2() {
        let inputs = [
            CuVector::<f32>::from_host_data(&[0.0, 0.0]),
            CuVector::<f32>::from_host_data(&[0.0, 1.0]),
            CuVector::<f32>::from_host_data(&[1.0, 0.0]),
            CuVector::<f32>::from_host_data(&[1.0, 1.0]),
        ];
        let ideals = [
            CuVector::<f32>::from_host_data(&[0.0]),
            CuVector::<f32>::from_host_data(&[1.0]),
            CuVector::<f32>::from_host_data(&[1.0]),
            CuVector::<f32>::from_host_data(&[0.0]),
        ];

        let network = NetworkBuilder::new_dense(2, false, Some(ActivationDescriptor::sigmoid()))
            .dense(3, false, Some(ActivationDescriptor::sigmoid()))
            .build(1);

        println!("Network descriptor :\n{:?}", network);

        let mut cuda = CudaHandleHolder::new(CurandRngType::PseudoDefault);
        let mut network = network.create_network();
        cuda.curand.generate_uniform_range(network.params_mut(), -1.0, 1.0, &DEFAULT_STREAM);

        let mut workspace = network.create_workspace();
        let mut training_space = network.create_training_space();
        let mut output_buffer = CuVector::<f32>::zero(1);
        let mut output_signal_buffer = CuVector::<f32>::zero(1);
        let mut weights_change_buffer = CuVector::<f32>::zero(network.params().len());

        for iter in 0..5000 {
            let mut error = 0.0;
            for i in 0..4 {
                network.forward_training(&mut cuda, &mut workspace, &inputs[i], &mut output_buffer);

                CuVectorMath::<f32>::sub(&output_buffer, &ideals[i], &mut output_signal_buffer, &DEFAULT_STREAM);
                error += cuda.cublas.asum(&output_signal_buffer);

                network.backpropagate(&mut cuda, 0.1, 1.0,
                                      &mut workspace, &inputs[i],
                                      &output_buffer, &mut output_signal_buffer,
                                      &mut training_space, &mut weights_change_buffer, None);

            }
            println!("Iteration {}, Error = {:.20}", iter, error);

            network.params_mut().add(&weights_change_buffer, &DEFAULT_STREAM);

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
            LayerDescriptor::dense(3, hidden_dimension, false, None)
        ).create_layer(weights1_holder.as_wrapped_ptr());
        let layer2 = Box::new(
            LayerDescriptor::dense(hidden_dimension+1, 1, false, None)
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