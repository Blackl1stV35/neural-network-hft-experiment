//! Window-based min-max scaler matching the Python implementation.
//!
//! Scales each feature independently within a rolling window,
//! making the model invariant to absolute price levels.

/// Rolling window min-max scaler.
pub struct WindowMinMaxScaler {
    window_size: usize,
    epsilon: f32,
}

impl WindowMinMaxScaler {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            epsilon: 1e-8,
        }
    }

    /// Scale a 2D feature matrix (n_samples × n_features).
    /// Each row is scaled using the min/max of its preceding window.
    pub fn transform(&self, data: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let n_samples = data.len();
        if n_samples == 0 {
            return vec![];
        }
        let n_features = data[0].len();
        let mut scaled = vec![vec![0.0f32; n_features]; n_samples];

        for i in 0..n_samples {
            let start = if i >= self.window_size { i - self.window_size + 1 } else { 0 };

            for f in 0..n_features {
                let mut w_min = f32::MAX;
                let mut w_max = f32::MIN;

                for j in start..=i {
                    let val = data[j][f];
                    if val < w_min { w_min = val; }
                    if val > w_max { w_max = val; }
                }

                let denom = w_max - w_min + self.epsilon;
                scaled[i][f] = (data[i][f] - w_min) / denom;
            }
        }

        scaled
    }

    /// Scale the last entry given the current window of data.
    /// More efficient for real-time use (only scales one row).
    pub fn transform_last(&self, window: &[Vec<f32>]) -> Vec<f32> {
        let n = window.len();
        if n == 0 {
            return vec![];
        }
        let n_features = window[0].len();
        let start = if n > self.window_size { n - self.window_size } else { 0 };
        let mut result = vec![0.0f32; n_features];

        for f in 0..n_features {
            let mut w_min = f32::MAX;
            let mut w_max = f32::MIN;

            for j in start..n {
                let val = window[j][f];
                if val < w_min { w_min = val; }
                if val > w_max { w_max = val; }
            }

            let denom = w_max - w_min + self.epsilon;
            result[f] = (window[n - 1][f] - w_min) / denom;
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_data() {
        let scaler = WindowMinMaxScaler::new(10);
        let data = vec![vec![100.0, 200.0]; 20];
        let scaled = scaler.transform(&data);

        // Constant data → should be ~0 (within epsilon)
        for row in &scaled {
            for &val in row {
                assert!(val.abs() < 0.01, "Expected ~0 for constant data, got {}", val);
            }
        }
    }

    #[test]
    fn test_increasing_data() {
        let scaler = WindowMinMaxScaler::new(5);
        let data: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32]).collect();
        let scaled = scaler.transform(&data);

        // Last element in window should be ~1.0
        for i in 5..10 {
            assert!(scaled[i][0] > 0.9, "Expected ~1.0 at end of window");
        }
    }

    #[test]
    fn test_transform_last() {
        let scaler = WindowMinMaxScaler::new(5);
        let window: Vec<Vec<f32>> = vec![
            vec![10.0], vec![12.0], vec![8.0], vec![15.0], vec![11.0],
        ];
        let result = scaler.transform_last(&window);

        // 11.0 in range [8, 15]: (11-8)/(15-8) ≈ 0.4286
        assert!((result[0] - 0.4286).abs() < 0.01);
    }

    #[test]
    fn test_price_shift_invariance() {
        let scaler = WindowMinMaxScaler::new(10);

        let data1: Vec<Vec<f32>> = (0..20).map(|i| vec![(i as f32) * 0.5]).collect();
        let data2: Vec<Vec<f32>> = (0..20).map(|i| vec![(i as f32) * 0.5 + 2000.0]).collect();

        let scaled1 = scaler.transform(&data1);
        let scaled2 = scaler.transform(&data2);

        for i in 0..20 {
            assert!(
                (scaled1[i][0] - scaled2[i][0]).abs() < 1e-4,
                "Scaler should be invariant to price shifts"
            );
        }
    }
}
