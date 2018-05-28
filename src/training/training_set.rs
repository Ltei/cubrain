
use cumath::*;




pub struct PackedTrainingSet {
    pub inputs: Vec<CuVector<f32>>,
    pub packed_outputs: CuVector<f32>,
}

pub struct TrainingSet {
    items: Vec<TrainingSetItem>
}
pub struct TrainingSetItem {
    input: Vec<f32>,
    output: Vec<f32>,
}
impl TrainingSet {
    pub fn new() -> TrainingSet {
        TrainingSet { items: Vec::new() }
    }
    pub fn add(&mut self, input: Vec<f32>, output: Vec<f32>) {
        if self.items.len() != 0 {
            assert_eq!(input.len(), self.items[0].input.len());
            assert_eq!(output.len(), self.items[0].output.len());
        }
        self.items.push(TrainingSetItem { input, output })
    }

    pub fn pack(&self) -> PackedTrainingSet {
        PackedTrainingSet {
            inputs: self.items.iter().map(|x| { CuVector::<f32>::from_host_data(&x.input) }).collect(),
            packed_outputs: {
                let mut vec = Vec::with_capacity(self.items.len() * self.items[0].output.len());
                self.items.iter().for_each(|x| vec.append(&mut x.output.to_owned()));
                CuVector::<f32>::from_host_data(&vec)
            },
        }
    }
}