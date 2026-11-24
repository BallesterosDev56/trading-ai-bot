use crate::book::{Data, OrderBook, Price, QTY_EPS, to_price};
use crate::feed::{AccountId, NewOrder, OrderId, Side};

/// A single executed match between a resting order and an incoming order.
#[derive(Debug, Clone)]
pub struct Trade {
    pub symbol: String,
    pub price: Price,
    pub quantity: f64,
    pub timestamp: u64,
    pub buy_order_id: OrderId,
    pub buy_account: AccountId,
    pub sell_order_id: OrderId,
    pub sell_account: AccountId,
    /// The side of the incoming order, i.e. who crossed the spread.
    pub aggressor: Side,
}

impl Trade {
    /// Builds the trade between the incoming order and a resting order it crossed with.
    fn new(incoming: &NewOrder, price: Price, resting: &Data) -> Self {
        let (buy_order_id, buy_account, sell_order_id, sell_account) = match incoming.side {
            Side::Buy => (incoming.id, incoming.account, resting.id, resting.account),
            Side::Sell => (resting.id, resting.account, incoming.id, incoming.account),
        };
        Trade {
            symbol: incoming.symbol.clone(),
            price,
            quantity: resting.quantity,
            timestamp: incoming.timestamp,
            buy_order_id,
            buy_account,
            sell_order_id,
            sell_account,
            aggressor: incoming.side,
        }
    }
}

/// Matches an incoming order against the resting orders on the opposite side
/// of the book for its symbol, at the best available price. Whenever the
/// incoming order's price crosses the best resting price, a trade is
/// executed: the resting order is either fully consumed (and removed) or
/// partially filled (and reduced in place), and the incoming order's
/// remaining quantity shrinks accordingly. Once no more crosses are
/// possible, any quantity left on the incoming order is rested in the book.
/// Returns the trades executed, in execution order.
pub fn match_order(book: &mut OrderBook, order: &NewOrder) -> Vec<Trade> {
    let incoming_price = to_price(order.price);
    let opposite_side = order.side.opposite();
    let mut remaining = order.quantity;
    let mut trades = Vec::new();

    while remaining > QTY_EPS {
        // 1. Stop when the other side of the book is empty.
        let Some(best_price) = book.best_price(&order.symbol, opposite_side) else {
            break;
        };

        // 2. Stop when the best resting price no longer crosses our limit.
        let crosses = match order.side {
            Side::Buy => incoming_price >= best_price,
            Side::Sell => incoming_price <= best_price,
        };
        if !crosses {
            break;
        }

        // 3. Fill against the best resting order and record the trade at its price.
        let Some((resting, _fully_filled)) =
            book.match_best(&order.symbol, opposite_side, remaining)
        else {
            break;
        };
        remaining -= resting.quantity;
        trades.push(Trade::new(order, best_price, &resting));
    }

    // 4. Whatever is left rests in the book at the limit price.
    if remaining > QTY_EPS {
        book.create_order(
            &order.symbol,
            order.side,
            order.price,
            Data {
                id: order.id,
                account: order.account,
                timestamp: order.timestamp,
                quantity: remaining,
            },
        );
    }
    trades
}

#[cfg(test)]
mod tests {
    use super::*;

    fn order(id: u64, side: Side, price: f64, quantity: f64) -> NewOrder {
        NewOrder {
            id,
            timestamp: 0,
            account: id as u32,
            symbol: "X".into(),
            side,
            price,
            quantity,
        }
    }

    #[test]
    fn non_crossing_order_rests() {
        let mut book = OrderBook::default();
        assert!(match_order(&mut book, &order(1, Side::Sell, 10.0, 2.0)).is_empty());
        assert!(match_order(&mut book, &order(2, Side::Buy, 9.0, 2.0)).is_empty());
        assert_eq!(book.best_price("X", Side::Buy), Some(900));
        assert_eq!(book.best_price("X", Side::Sell), Some(1000));
    }

    #[test]
    fn crossing_order_trades_at_resting_price_and_rests_remainder() {
        let mut book = OrderBook::default();
        match_order(&mut book, &order(1, Side::Sell, 10.0, 2.0));
        let trades = match_order(&mut book, &order(2, Side::Buy, 10.5, 5.0));
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].price, 1000);
        assert_eq!(trades[0].quantity, 2.0);
        assert_eq!(trades[0].aggressor, Side::Buy);
        assert_eq!((trades[0].buy_order_id, trades[0].sell_order_id), (2, 1));
        // 3.0 left over rests as the new best bid at the limit price.
        assert_eq!(book.best_price("X", Side::Buy), Some(1050));
        assert_eq!(book.best_price("X", Side::Sell), None);
    }

    #[test]
    fn sweeps_multiple_levels_best_first() {
        let mut book = OrderBook::default();
        match_order(&mut book, &order(1, Side::Sell, 10.0, 1.0));
        match_order(&mut book, &order(2, Side::Sell, 10.1, 1.0));
        let trades = match_order(&mut book, &order(3, Side::Buy, 10.1, 2.0));
        let prices: Vec<_> = trades.iter().map(|t| t.price).collect();
        assert_eq!(prices, vec![1000, 1010]);
    }

    #[test]
    fn float_residue_does_not_rest() {
        let mut book = OrderBook::default();
        match_order(&mut book, &order(1, Side::Sell, 10.0, 0.1));
        match_order(&mut book, &order(2, Side::Sell, 10.0, 0.2));
        // 0.1 + 0.2 is not exactly 0.3 in floating point; no dust should be left behind.
        match_order(&mut book, &order(3, Side::Buy, 10.0, 0.3));
        assert_eq!(book.best_price("X", Side::Buy), None);
        assert_eq!(book.best_price("X", Side::Sell), None);
    }
}
