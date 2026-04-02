//! Lock-free ring buffer for tick data ingestion.
//!
//! Pre-allocates fixed-size arrays to avoid allocations in the hot path.

/// Fixed-size ring buffer for f32 feature vectors.
pub struct RingBuffer {
    data: Vec<Vec<f32>>,
    capacity: usize,
    feature_dim: usize,
    head: usize,
    len: usize,
}

impl RingBuffer {
    /// Create a new ring buffer.
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of entries
    /// * `feature_dim` - Dimension of each feature vector
    pub fn new(capacity: usize, feature_dim: usize) -> Self {
        let data = vec![vec![0.0f32; feature_dim]; capacity];
        Self {
            data,
            capacity,
            feature_dim,
            head: 0,
            len: 0,
        }
    }

    /// Push a new feature vector into the buffer.
    pub fn push(&mut self, features: &[f32]) {
        assert_eq!(features.len(), self.feature_dim,
            "Feature dim mismatch: expected {}, got {}",
            self.feature_dim, features.len());

        self.data[self.head].copy_from_slice(features);
        self.head = (self.head + 1) % self.capacity;
        self.len = (self.len + 1).min(self.capacity);
    }

    /// Get the last N entries as a contiguous 2D array.
    ///
    /// Returns None if fewer than `n` entries have been pushed.
    pub fn last_n(&self, n: usize) -> Option<Vec<Vec<f32>>> {
        if n > self.len {
            return None;
        }

        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let idx = (self.head + self.capacity - n + i) % self.capacity;
            result.push(self.data[idx].clone());
        }
        Some(result)
    }

    /// Get the last N entries as a flat f32 slice (row-major).
    pub fn last_n_flat(&self, n: usize) -> Option<Vec<f32>> {
        let rows = self.last_n(n)?;
        let mut flat = Vec::with_capacity(n * self.feature_dim);
        for row in rows {
            flat.extend_from_slice(&row);
        }
        Some(flat)
    }

    /// Number of entries currently in the buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether the buffer is full.
    pub fn is_full(&self) -> bool {
        self.len == self.capacity
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_and_retrieve() {
        let mut buf = RingBuffer::new(5, 3);
        buf.push(&[1.0, 2.0, 3.0]);
        buf.push(&[4.0, 5.0, 6.0]);
        buf.push(&[7.0, 8.0, 9.0]);

        assert_eq!(buf.len(), 3);

        let last2 = buf.last_n(2).unwrap();
        assert_eq!(last2[0], vec![4.0, 5.0, 6.0]);
        assert_eq!(last2[1], vec![7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_overflow() {
        let mut buf = RingBuffer::new(3, 2);
        for i in 0..5 {
            buf.push(&[i as f32, (i * 10) as f32]);
        }

        assert_eq!(buf.len(), 3);

        let all = buf.last_n(3).unwrap();
        assert_eq!(all[0], vec![2.0, 20.0]);
        assert_eq!(all[1], vec![3.0, 30.0]);
        assert_eq!(all[2], vec![4.0, 40.0]);
    }

    #[test]
    fn test_insufficient_data() {
        let mut buf = RingBuffer::new(10, 4);
        buf.push(&[1.0, 2.0, 3.0, 4.0]);

        assert!(buf.last_n(5).is_none());
        assert!(buf.last_n(1).is_some());
    }

    #[test]
    fn test_flat_output() {
        let mut buf = RingBuffer::new(5, 2);
        buf.push(&[1.0, 2.0]);
        buf.push(&[3.0, 4.0]);

        let flat = buf.last_n_flat(2).unwrap();
        assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
