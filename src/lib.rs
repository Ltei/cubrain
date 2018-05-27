#![allow(dead_code)]

extern crate cumath;
extern crate cumath_nn;

mod meta;
pub use self::meta::*;

mod network;
pub use self::network::*;

mod training_set;
pub use self::training_set::*;

mod error_calculator;
pub use self::error_calculator::*;

mod magnitude_manager;
pub mod training;



use cumath::*;
use cumath_nn::*;



pub struct CudaHandleHolder {
    cublas: Cublas,
    curand: CurandGenerator,
    cudnn: Cudnn,
}
impl CudaHandleHolder {
    pub fn new(rand: CurandRngType) -> CudaHandleHolder {
        CudaHandleHolder {
            cublas: Cublas::new().unwrap(),
            curand: CurandGenerator::new(rand).unwrap(),
            cudnn: Cudnn::new(),
        }
    }
    pub fn cublas(&self) -> &Cublas { &self.cublas }
    pub fn cublas_mut(&mut self) -> &mut Cublas { &mut self.cublas }
    pub fn curand(&self) -> &CurandGenerator { &self.curand }
    pub fn curand_mut(&mut self) -> &mut CurandGenerator { &mut self.curand }
    pub fn cudnn(&self) -> &Cudnn { &self.cudnn }
    pub fn cudnn_mut(&mut self) -> &mut Cudnn { &mut self.cudnn }
}


pub trait CloneStructure {
    fn clone_structure(&self) -> Self;
}
pub trait GetParams {
    fn params(&self) -> &CuVectorDeref<f32>;
    fn params_mut(&mut self) -> &mut CuVectorDeref<f32>;
}
pub trait ForwardInference {
    fn workspace_len(&self) -> usize;
    fn forward_inference(&self, cuda: &mut CudaHandleHolder, input: &CuVectorDeref<f32>, output: &mut CuVectorDeref<f32>, workspace: &mut CuVectorDeref<f32>);
}