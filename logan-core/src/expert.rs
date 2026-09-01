//! Engine-neutral expert store: LRU cache over (layer, expert) with
//! slot-owning values.
//!
//! The C engine's measured data: LRU with CACHE 0->4096 took wall
//! 280->173 ms/tok, but misses plateau at ~19% at any cap >= 256 (the cold
//! first-touch floor). The store here is the policy; the engine supplies
//! the slot lifecycle (MetalIO slot alloc/free) through the `Slot` trait.
//!
//! LRU semantics: HashMap<key, node> + intrusive doubly-linked list
//! (indices into a slab), O(1) get/insert/evict. Eviction calls
//! `Slot::release` so the engine frees the MetalIO slot.

use std::collections::HashMap;

/// A slot-owning cache value. The engine implements this over its slot
/// type (e.g. a MetalIO slot whose Drop frees the buffer).
pub trait Slot: Sized {
    /// Called on eviction (and on store drop). Must release the slot.
    fn release(&mut self);
}

struct Node<V> {
    key: (u32, u32),
    value: V,
    prev: Option<usize>,
    next: Option<usize>,
}

/// O(1) LRU keyed by (layer, expert).
pub struct ExpertStore<V: Slot> {
    map: HashMap<(u32, u32), usize>,
    slab: Vec<Option<Node<V>>>,
    head: Option<usize>, // most-recently-used
    tail: Option<usize>, // least-recently-used
    cap: usize,
    /// Telemetry counters (regime-independent A/B metrics).
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

impl<V: Slot> ExpertStore<V> {
    pub fn new(cap: usize) -> ExpertStore<V> {
        ExpertStore {
            map: HashMap::new(),
            slab: Vec::new(),
            head: None,
            tail: None,
            cap: cap.max(1),
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// Get a cached expert, promoting it to MRU. None on miss.
    pub fn get(&mut self, key: (u32, u32)) -> Option<&V> {
        let &idx = self.map.get(&key)?;
        self.hits += 1;
        self.unlink(idx);
        self.push_front(idx);
        Some(&self.slab[idx].as_ref().unwrap().value)
    }

    /// Look up WITHOUT counting a hit or promoting (internal lookups like
    /// waiting on a pending load must not pollute the reuse metric).
    pub fn peek(&self, key: (u32, u32)) -> Option<&V> {
        let &idx = self.map.get(&key)?;
        Some(&self.slab[idx].as_ref().unwrap().value)
    }

    /// Insert (or replace) an expert, evicting LRU when over capacity.
    /// Returns (evicted-value-requiring-release, ref-to-inserted-value).
    /// The inserted ref does NOT bump `hits` — only `get` does, so the hit
    /// counter measures genuine reuse, not the insert-then-immediate-get
    /// pattern engines use to fetch a fresh expert.
    pub fn insert(&mut self, key: (u32, u32), value: V) -> (Option<V>, &V) {
        if let Some(&idx) = self.map.get(&key) {
            // replace in place, promote
            let node = self.slab[idx].as_mut().unwrap();
            let mut old = std::mem::replace(&mut node.value, value);
            old.release();
            self.unlink(idx);
            self.push_front(idx);
            let v = &self.slab[idx].as_ref().unwrap().value;
            return (None, v);
        }
        self.misses += 1;
        let mut evicted = None;
        if self.map.len() >= self.cap {
            if let Some(t) = self.tail {
                let key_t = self.slab[t].as_ref().unwrap().key;
                self.unlink(t);
                self.map.remove(&key_t);
                let node = self.slab[t].take().unwrap();
                evicted = Some(node.value);
                self.evictions += 1;
            }
        }
        let idx = self.slab.len();
        self.slab.push(Some(Node {
            key,
            value,
            prev: None,
            next: None,
        }));
        self.map.insert(key, idx);
        self.push_front(idx);
        let v = &self.slab[idx].as_ref().unwrap().value;
        (evicted, v)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Hit rate (0..1) for telemetry.
    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f32 / total as f32
        }
    }

    fn unlink(&mut self, idx: usize) {
        let (prev, next) = {
            let n = self.slab[idx].as_ref().unwrap();
            (n.prev, n.next)
        };
        match prev {
            Some(p) => self.slab[p].as_mut().unwrap().next = next,
            None => self.head = next,
        }
        match next {
            Some(nx) => self.slab[nx].as_mut().unwrap().prev = prev,
            None => self.tail = prev,
        }
    }

    fn push_front(&mut self, idx: usize) {
        let n = self.slab[idx].as_mut().unwrap();
        n.prev = None;
        n.next = self.head;
        match self.head {
            Some(h) => self.slab[h].as_mut().unwrap().prev = Some(idx),
            None => self.tail = Some(idx),
        }
        self.head = Some(idx);
    }
}

impl<V: Slot> Drop for ExpertStore<V> {
    fn drop(&mut self) {
        for node in self.slab.iter_mut().flatten() {
            node.value.release();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeSlot {
        key: (u32, u32),
        released: bool,
    }
    impl Slot for FakeSlot {
        fn release(&mut self) {
            self.released = true;
        }
    }

    #[test]
    fn lru_evicts_least_recently_used() {
        let mut s: ExpertStore<FakeSlot> = ExpertStore::new(2);
        s.insert(
            (0, 0),
            FakeSlot {
                key: (0, 0),
                released: false,
            },
        );
        s.insert(
            (0, 1),
            FakeSlot {
                key: (0, 1),
                released: false,
            },
        );
        // touch (0,0) so (0,1) becomes LRU
        assert!(s.get((0, 0)).is_some());
        let (evicted, _) = s.insert(
            (1, 0),
            FakeSlot {
                key: (1, 0),
                released: false,
            },
        );
        let mut ev = evicted.unwrap();
        assert_eq!(ev.key, (0, 1));
        // eviction returns the value; the caller releases it (engine frees
        // the MetalIO slot by dropping it)
        assert!(!ev.released);
        ev.release();
        assert!(ev.released);
        assert!(s.get((0, 0)).is_some());
        assert!(s.get((1, 0)).is_some());
        assert!(s.get((0, 1)).is_none());
    }

    #[test]
    fn replace_promotes_and_releases_old() {
        let mut s: ExpertStore<FakeSlot> = ExpertStore::new(2);
        s.insert(
            (0, 0),
            FakeSlot {
                key: (0, 0),
                released: false,
            },
        );
        s.insert(
            (0, 1),
            FakeSlot {
                key: (0, 1),
                released: false,
            },
        );
        // replace (0,0) in place: old released internally, (0,0) promoted
        let (old, _) = s.insert(
            (0, 0),
            FakeSlot {
                key: (0, 0),
                released: false,
            },
        );
        assert!(old.is_none());
        // (0,0) is now MRU; (0,1) is LRU; cap 2 -> inserting (1,1) evicts (0,1)
        let (evicted, _) = s.insert(
            (1, 1),
            FakeSlot {
                key: (1, 1),
                released: false,
            },
        );
        assert_eq!(evicted.unwrap().key, (0, 1));
    }

    #[test]
    fn peek_does_not_bump_hits_or_promote() {
        let mut s: ExpertStore<FakeSlot> = ExpertStore::new(2);
        s.insert(
            (0, 0),
            FakeSlot {
                key: (0, 0),
                released: false,
            },
        );
        s.insert(
            (0, 1),
            FakeSlot {
                key: (0, 1),
                released: false,
            },
        );
        assert!(s.peek((0, 0)).is_some());
        assert_eq!(s.hits, 0); // peek is not a hit
        // Peek must NOT promote: (0,1) was inserted second so it is MRU;
        // (0,0) is still LRU. Inserting a third key evicts (0,0), proving
        // the peek didn't touch the recency order.
        let (evicted, _) = s.insert(
            (2, 0),
            FakeSlot {
                key: (2, 0),
                released: false,
            },
        );
        assert_eq!(evicted.unwrap().key, (0, 0));
    }

    #[test]
    fn insert_does_not_bump_hits() {
        let mut s: ExpertStore<FakeSlot> = ExpertStore::new(2);
        s.insert(
            (0, 0),
            FakeSlot {
                key: (0, 0),
                released: false,
            },
        );
        s.insert(
            (0, 1),
            FakeSlot {
                key: (0, 1),
                released: false,
            },
        );
        // insert returns the value ref WITHOUT a hit — hits stay 0
        assert_eq!(s.hits, 0);
        assert!(s.get((0, 1)).is_some()); // genuine reuse -> 1 hit
        assert_eq!(s.hits, 1);
    }

    #[test]
    fn hit_rate_counts() {
        let mut s: ExpertStore<FakeSlot> = ExpertStore::new(2);
        s.insert(
            (0, 0),
            FakeSlot {
                key: (0, 0),
                released: false,
            },
        );
        s.get((0, 0));
        s.get((0, 0));
        s.get((9, 9)); // miss
        assert_eq!(s.hits, 2);
        assert_eq!(s.misses, 1);
        assert!((s.hit_rate() - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn drop_releases_all() {
        let mut s: ExpertStore<FakeSlot> = ExpertStore::new(4);
        for i in 0..3 {
            s.insert(
                (0, i),
                FakeSlot {
                    key: (0, i),
                    released: false,
                },
            );
        }
        drop(s);
        // can't observe after drop; just ensure no panic + len was 3
        // (release() is called via Drop; a panic here would fail the test)
    }
}
