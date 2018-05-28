
pub mod layer_dense;
pub mod layer_convolution;
pub mod layer_descriptor;
pub mod network;
pub mod network_builder;
pub mod layer_activation;

pub use self::layer_dense::*;
pub use self::layer_convolution::*;
pub use self::layer_descriptor::*;
pub use self::network::*;
pub use self::network_builder::*;
pub use self::layer_activation::*;



use cumath::*;
use CudaHandleHolder;
use ForwardInference;


pub trait Layer: ForwardInference {

    fn clone_structure(&self, weights: CuVectorPtr<f32>) -> Box<Layer>;

    fn params_count(&self) -> usize;

    fn forward_training(&self, cuda: &mut CudaHandleHolder, input: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>);

    fn backward_training(&self, cuda: &mut CudaHandleHolder, learning_rate: f32, momentum: f32,
                         layer_input: &CuVectorDeref<f32>, layer_output: &mut CuVectorDeref<f32>, front_signal: &CuVectorDeref<f32>,
                         weights_change: &mut CuVectorDeref<f32>, back_signal: Option<&mut CuVectorDeref<f32>>);

}
