
use cumath_nn::*;
use super::*;
use NetworkDescriptor;
use LayerDescriptor;


mod pending_layer {
    use super::*;

    pub struct NetworkBuilder {
        pub(super) layers: Vec<LayerDescriptor>,
        pub(super) input_len: usize,
        pub(super) activations: Vec<(CudnnActivationMode, f64)>,
    }
    impl NetworkBuilder {
        pub fn dense(mut self, input_len: usize) -> NetworkBuilder {
            self.layers.push(LayerDescriptor::dense(self.input_len, input_len));

            for act in &self.activations {
                self.layers.push(LayerDescriptor::activation(act.0, input_len, act.1));
            }
            self.activations.clear();

            self.input_len = input_len;
            self
        }
        pub fn activation(mut self, mode: CudnnActivationMode, coef: f64) -> NetworkBuilder {
            self.activations.push((mode, coef));
            self
        }
        pub fn convolution(mut self, input_rows: usize, input_cols: usize, kernels: Vec<ConvolutionKernelDescriptor>) -> super::NetworkBuilder {
            let input_len = input_rows * input_cols;
            self.layers.push(LayerDescriptor::dense(self.input_len, input_len));

            for act in &self.activations {
                self.layers.push(LayerDescriptor::activation(act.0, input_len, act.1));
            }
            self.activations.clear();

            self.layers.push(LayerDescriptor::convolution(input_rows, input_cols, kernels));
            super::NetworkBuilder { layers: self.layers }
        }
        pub fn build(mut self, output_len: usize) -> NetworkDescriptor {
            self.layers.push(LayerDescriptor::dense(self.input_len, output_len));

            for act in &self.activations {
                self.layers.push(LayerDescriptor::activation(act.0, output_len, act.1));
            }
            self.activations.clear();

            NetworkDescriptor { layers: self.layers }
        }
    }
}


pub struct NetworkBuilder {
    layers: Vec<LayerDescriptor>,
}
impl NetworkBuilder {

    pub fn new_dense(input_len: usize) -> pending_layer::NetworkBuilder {
        pending_layer::NetworkBuilder { layers: Vec::new(), input_len, activations: Vec::new() }
    }
    pub fn new_convolution(input_rows: usize, input_cols: usize, kernels: Vec<ConvolutionKernelDescriptor>) -> NetworkBuilder {
        NetworkBuilder {
            layers: vec![LayerDescriptor::convolution(input_rows, input_cols, kernels)]
        }
    }
    pub fn dense(self) -> pending_layer::NetworkBuilder {
        pending_layer::NetworkBuilder {
            input_len: self.layers.iter().last().unwrap().output_len(),
            layers: self.layers,
            activations: Vec::new(),
        }
    }
    pub fn activation(mut self, mode: CudnnActivationMode, coef: f64) -> NetworkBuilder {
        let len = self.layers.last().unwrap().output_len();
        self.layers.push(LayerDescriptor::activation(mode, len, coef));
        self
    }
    pub fn convolution(mut self, input_rows: usize, input_cols: usize, kernels: Vec<ConvolutionKernelDescriptor>) -> NetworkBuilder {
        assert_eq!(input_rows*input_cols, self.layers.last().unwrap().output_len());
        self.layers.push(LayerDescriptor::convolution(input_rows, input_cols, kernels));
        self
    }
    pub fn build(self) -> NetworkDescriptor {
        NetworkDescriptor { layers: self.layers }
    }
}





#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn build() {

        let network = NetworkBuilder::new_dense(50)
            .activation(CudnnActivationMode::Sigmoid, 2.5)
            .convolution(5, 7,
                         vec![
                             ConvolutionKernelDescriptor::new(3, 3, 1, 1),
                             ConvolutionKernelDescriptor::new(2, 4, 1, 1),
                             ConvolutionKernelDescriptor::new(2, 1, 1, 1),
                         ]).dense().activation(CudnnActivationMode::Tanh, 7.2).build(60);

        assert_eq!(network.input_len(), 50);
        assert_eq!(network.output_len(), 60);
        assert_eq!(network.layers.len(), 5);
        assert_eq!(network.input_len(), network.layers[0].input_len());
        assert_eq!(network.output_len(), network.layers[4].output_len());
        for i in 0..4 {
            assert_eq!(network.layers[i].output_len(), network.layers[i+1].input_len());
        }
    }

}
