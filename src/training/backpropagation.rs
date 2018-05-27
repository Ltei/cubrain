

use std;

use training_set::*;
use CesureNetwork;
use CudaHandleHolder;






pub fn train <CesureNetworkT: CesureNetwork>
        (cuda: &mut CudaHandleHolder, input_cesure: &mut CesureNetworkT, training_set: &PackedTrainingSet,
         learning_rate: f32, momentum: f32, nb_iterations: usize, print_every: usize) {

    for iter in 0..nb_iterations {



    }

 }