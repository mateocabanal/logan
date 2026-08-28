//! Typed monotonic identifiers. The scheduler core is single-owner, so
//! allocation is a plain counter (no atomics, no locks — the owner is the
//! only thread that ever calls these).

use serde::{Deserialize, Serialize};
use std::fmt;

macro_rules! typed_id {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(
            Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
        )]
        pub struct $name(pub(crate) u64);

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!(stringify!($name), "({})"), self.0)
            }
        }
    };
}

typed_id!(
    SessionId,
    "A session: one engine sequence of steps (e.g. one decode)."
);
typed_id!(
    Ticket,
    "One submitted action; affects live state at most once."
);
typed_id!(
    LoadId,
    "One physical expert/resource load key (residency #47)."
);
typed_id!(
    Generation,
    "A resource generation; in-flight work pins exact versions."
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_distinct_and_comparable() {
        let a = SessionId(1);
        let b = SessionId(2);
        assert_ne!(a, b);
        assert!(a < b);
        assert_eq!(a, SessionId(1));
        assert_eq!(format!("{a}"), "SessionId(1)");
    }
}
