

use CudaHandleHolder;
use GetParams;
use CloneStructure;
use VectorUtils;
use error_calculator::*;
use magnitude_manager::*;



pub fn train<T: GetParams + CloneStructure>(cuda: &mut CudaHandleHolder, cesure: &mut T,
                           error_calculation: &mut ErrorCalculator<T>, initial_magnitude: f32,
                           nb_iterations: usize, psychose: usize, print_every: usize) {

    let magnitude_manager = LinearMagnitudeManager::new(nb_iterations, initial_magnitude, 0.0);

    let mut best_cesure = cesure.clone_structure();
    best_cesure.params_mut().clone_from_device(cesure.params());
    let mut best_error = error_calculation.compute_error(&best_cesure, cuda);

    let mut tmp_cesure = cesure.clone_structure();

    let mut iterations_since_change = 0;

    for iteration in 0..nb_iterations {

        let magnitude = magnitude_manager.get_magnitude(iteration);

        if iterations_since_change >= psychose {
            tmp_cesure.params_mut().clone_randomized_from(cuda, best_cesure.params(), (initial_magnitude*magnitude)/(1000.0*magnitude));
            let tmp_error = error_calculation.compute_error(&tmp_cesure, cuda);
            let tmp = best_cesure;
            best_cesure = tmp_cesure;
            tmp_cesure = tmp;
            best_error = tmp_error;
            iterations_since_change = 0;
        } else {
            tmp_cesure.params_mut().clone_randomized_from(cuda, best_cesure.params(), magnitude);
            let tmp_error = error_calculation.compute_error(&tmp_cesure, cuda);

            if tmp_error <= best_error {
                let tmp = best_cesure;
                best_cesure = tmp_cesure;
                tmp_cesure = tmp;
                best_error = tmp_error;
                iterations_since_change = 0;
            } else {
                iterations_since_change += 1;
            }
        }


        if iteration % print_every == 0 {
            println!("Iteration {}, Error = {}, Magnitude = {}", iteration, best_error, magnitude);
        }

    }

    cesure.params_mut().clone_from_device(best_cesure.params());

}