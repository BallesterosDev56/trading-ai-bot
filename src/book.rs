use crate::feed::{AccountId, OrderId, Side};
use std::collections::{BTreeMap, HashMap, VecDeque};

/// A price represented as an integer number of ticks, so it can be used as
/// an ordered `BTreeMap` key without the pitfalls of comparing `f64` (NaN,
/// rounding drift). One tick is one hundredth of a unit (i.e. cents).
pub type Price = i64;

/// Converts a raw price (as seen on the feed) into ticks.
pub fn to_price(price: f64) -> Price {
    (price * 100.0).round() as Price
}

/// Converts ticks back into a raw price.
pub fn from_price(price: Price) -> f64 {
    price as f64 / 100.0
}

/// Quantities below this are treated as zero, so float residue never rests in the book.
pub const QTY_EPS: f64 = 1e-9;

/// A single resting order at a given price level.
#[derive(Debug, Clone)]
pub struct Data {
    pub id: OrderId,
    pub account: AccountId,
    pub timestamp: u64,
    pub quantity: f64,
}

/// The resting bids and asks for a single symbol, keyed by price level.
#[derive(Debug, Clone, Default)]
pub struct Order {
    pub bids: BTreeMap<Price, VecDeque<Data>>,
    pub asks: BTreeMap<Price, VecDeque<Data>>,
}

impl Order {
    /// Highest resting bid price, if any.
    pub fn best_bid(&self) -> Option<Price> {
        self.bids.keys().next_back().copied()
    }

    /// Lowest resting ask price, if any.
    pub fn best_ask(&self) -> Option<Price> {
        self.asks.keys().next().copied()
    }
}

/// Where a resting order lives, so it can be found again by ID alone.
#[derive(Debug, Clone)]
struct IndexEntry {
    symbol: String,
    side: Side,
    price: Price,
}

/// The full order book across all symbols.
#[derive(Debug, Clone, Default)]
pub struct OrderBook {
    pub order_book: HashMap<String, Order>,
    /// Indexes every resting order by ID so `remove_order` doesn't need to
    /// know the symbol, side, or price to cancel it.
    index: HashMap<OrderId, IndexEntry>,
}

impl OrderBook {
    /// Inserts a new resting order into the book.
    pub fn create_order(&mut self, symbol: &str, side: Side, price: f64, data: Data) {
        let price = to_price(price);
        let order = self.order_book.entry(symbol.to_string()).or_default();
        let levels = match side {
            Side::Buy => &mut order.bids,
            Side::Sell => &mut order.asks,
        };

        self.index.insert(
            data.id,
            IndexEntry {
                symbol: symbol.to_string(),
                side,
                price,
            },
        );
        levels.entry(price).or_default().push_back(data);
    }

    /// Removes a resting order by ID, wherever it lives in the book.
    /// Returns the symbol it was resting on along with its data.
    pub fn remove_order(&mut self, id: OrderId) -> Option<(String, Data)> {
        let entry = self.index.remove(&id)?;
        let order = self.order_book.get_mut(&entry.symbol)?;
        let levels = match entry.side {
            Side::Buy => &mut order.bids,
            Side::Sell => &mut order.asks,
        };

        let level = levels.get_mut(&entry.price)?;
        let pos = level.iter().position(|data| data.id == id)?;
        let data = level.remove(pos)?;
        if level.is_empty() {
            levels.remove(&entry.price);
        }
        Some((entry.symbol, data))
    }

    /// Returns the best resting price for a symbol and side, if any orders
    /// are resting there (highest price for bids, lowest for asks).
    pub fn best_price(&self, symbol: &str, side: Side) -> Option<Price> {
        let order = self.order_book.get(symbol)?;
        match side {
            Side::Buy => order.bids.keys().next_back().copied(),
            Side::Sell => order.asks.keys().next().copied(),
        }
    }

    /// Fills up to `quantity` against the best resting order on `side` for
    /// `symbol`. Returns the resting order's original data with its
    /// `quantity` set to however much of it was actually filled, along with
    /// whether the resting order was fully consumed (and thus removed from
    /// the book) or only partially reduced.
    pub fn match_best(&mut self, symbol: &str, side: Side, quantity: f64) -> Option<(Data, bool)> {
        let order = self.order_book.get_mut(symbol)?;
        let levels = match side {
            Side::Buy => &mut order.bids,
            Side::Sell => &mut order.asks,
        };

        let price = match side {
            Side::Buy => *levels.keys().next_back()?,
            Side::Sell => *levels.keys().next()?,
        };

        let level = levels.get_mut(&price)?;
        let resting = level.front_mut()?;
        let filled = quantity.min(resting.quantity);
        resting.quantity -= filled;

        let mut filled_data = resting.clone();
        filled_data.quantity = filled;
        let fully_filled = resting.quantity <= QTY_EPS;

        if fully_filled {
            level.pop_front();
            if level.is_empty() {
                levels.remove(&price);
            }
            self.index.remove(&filled_data.id);
        }

        Some((filled_data, fully_filled))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(id: u64, quantity: f64) -> Data {
        Data {
            id,
            account: 0,
            timestamp: 0,
            quantity,
        }
    }

    #[test]
    fn best_price_picks_highest_bid_and_lowest_ask() {
        let mut book = OrderBook::default();
        book.create_order("X", Side::Buy, 9.0, data(1, 1.0));
        book.create_order("X", Side::Buy, 9.5, data(2, 1.0));
        book.create_order("X", Side::Sell, 10.5, data(3, 1.0));
        book.create_order("X", Side::Sell, 10.0, data(4, 1.0));
        assert_eq!(book.best_price("X", Side::Buy), Some(950));
        assert_eq!(book.best_price("X", Side::Sell), Some(1000));
    }

    #[test]
    fn remove_order_returns_symbol_and_cleans_empty_levels() {
        let mut book = OrderBook::default();
        book.create_order("X", Side::Buy, 9.0, data(1, 1.0));
        let (symbol, removed) = book.remove_order(1).unwrap();
        assert_eq!(symbol, "X");
        assert_eq!(removed.id, 1);
        assert!(book.order_book["X"].bids.is_empty());
        assert!(book.remove_order(1).is_none());
    }

    #[test]
    fn match_best_partially_fills_and_keeps_time_priority() {
        let mut book = OrderBook::default();
        book.create_order("X", Side::Sell, 10.0, data(1, 5.0));
        book.create_order("X", Side::Sell, 10.0, data(2, 5.0));
        let (filled, full) = book.match_best("X", Side::Sell, 2.0).unwrap();
        assert_eq!((filled.id, filled.quantity, full), (1, 2.0, false));
        let (filled, full) = book.match_best("X", Side::Sell, 10.0).unwrap();
        assert_eq!((filled.id, filled.quantity, full), (1, 3.0, true));
    }
}
