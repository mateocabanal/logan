//! Page abstraction for sparse paged state management.

/// A page of state data. Pages can be:
/// - resident in RAM
/// - backed by SSD (via snapshot/persistence)
/// - shared between sessions (COW)
#[derive(Debug, Clone)]
pub struct StatePage {
    /// Unique page identifier
    pub id: PageId,
    /// The page data
    pub data: Vec<f32>,
    /// Whether this page is pinned (not eligible for eviction)
    pub pinned: bool,
    /// Generation when this page was created
    pub generation: u64,
    /// Number of references to this page
    pub refs: usize,
}

/// Page identifier: maps to a position in the state
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PageId {
    /// Logical page index
    pub index: usize,
    /// Sub-page within a layer (for multi-head state)
    pub sub: usize,
}

impl PageId {
    pub fn new(index: usize, sub: usize) -> Self {
        PageId { index, sub }
    }
}

impl StatePage {
    /// Create a new page with the given capacity
    pub fn new(id: PageId, capacity: usize, generation: u64) -> Self {
        StatePage {
            id,
            data: vec![0.0; capacity],
            pinned: false,
            generation,
            refs: 1,
        }
    }

    /// Whether this page is empty (all zeros)
    pub fn is_empty(&self) -> bool {
        self.data.iter().all(|&x| x == 0.0)
    }

    /// Whether this page is a candidate for eviction
    pub fn is_evictable(&self) -> bool {
        !self.pinned && self.refs == 1
    }

    /// Pin this page (prevents eviction)
    pub fn pin(&mut self) {
        self.pinned = true;
    }

    /// Unpin this page (allows eviction)
    pub fn unpin(&mut self) {
        self.pinned = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_creation() {
        let id = PageId::new(0, 0);
        let page = StatePage::new(id, 256, 1);
        assert_eq!(page.data.len(), 256);
        assert!(!page.pinned);
        assert_eq!(page.refs, 1);
        assert!(page.is_empty());
    }

    #[test]
    fn test_page_pin_unpin() {
        let mut page = StatePage::new(PageId::new(0, 0), 64, 1);
        assert!(!page.is_evictable()); // refs=1, but pinned=false so evictable
        page.pin();
        assert!(!page.is_evictable());
        page.unpin();
        assert!(page.is_evictable());
    }

    #[test]
    fn test_page_duplicate_cow() {
        let mut page = StatePage::new(PageId::new(0, 0), 128, 1);
        page.refs = 2;
        assert!(!page.is_evictable()); // shared, cannot evict
        page.refs -= 1;
        assert!(page.is_evictable());
    }
}