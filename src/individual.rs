use rand::Rng;

use crate::summary::Summary;

#[derive(Debug, Clone)]
pub struct Individual {
    genome: Vec<bool>,
    summary: Option<Summary>,
}

impl Individual {
    pub fn new(genome: Vec<bool>) -> Self {
        Self {
            genome,
            summary: None,
        }
    }

    pub fn new_random(size: usize, rng: &mut impl Rng) -> Self {
        let genome = (0..size).map(|_| rng.gen()).collect();
        Self::new(genome)
    }
}

impl Individual {
    pub fn bitstring(&self) -> String {
        self.genome
            .iter()
            .map(|&b| if b { '1' } else { '0' })
            .collect()
    }

    pub fn summary(&self) -> Summary {
        self.summary.unwrap()
    }

    pub fn set_summary(&mut self, summary: Summary) {
        self.summary = Some(summary);
    }

    pub fn evaluate(&mut self) -> Summary {
        *self.summary.get_or_insert_with(|| {
            let count = self.genome.iter().filter(|x| **x).count();
            Summary { count }
        })
    }

    pub fn mutate(&mut self, rng: &mut impl Rng) {
        let n = self.genome.len();
        let p = 1.0 / (n as f64);
        for i in 0..n {
            if rng.gen_bool(p) {
                self.genome[i] = !self.genome[i];
            }
        }
        self.summary = None;
    }

    pub fn two_point_crossover(
        &self,
        other: &Individual,
        rng: &mut impl Rng,
    ) -> (Individual, Individual) {
        let mut child1 = self.genome.clone();
        let mut child2 = other.genome.clone();

        let left = rng.gen_range(0..self.genome.len());
        let right = rng.gen_range(left..=self.genome.len());

        for i in left..right {
            std::mem::swap(&mut child1[i], &mut child2[i]);
        }

        (Individual::new(child1), Individual::new(child2))
    }
}
