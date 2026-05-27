from client import trading_client


def get_account():
    account = trading_client.get_account()
    print(f"Account Status:    {account.status}")
    print(f"Buying Power:      ${float(account.buying_power):,.2f}")
    print(f"Portfolio Value:   ${float(account.portfolio_value):,.2f}")
    print(f"Cash:              ${float(account.cash):,.2f}")
    print(f"Equity:            ${float(account.equity):,.2f}")
    print(f"Day Trade Count:   {account.daytrade_count}")
    print(f"Pattern Day Trader:{account.pattern_day_trader}")
    return account


def get_positions():
    positions = trading_client.get_all_positions()
    if not positions:
        print("No open positions.")
        return positions
    print(f"\n{'Ticker':<8} {'Qty':<8} {'Avg Entry':<12} {'Market Val':<14} {'P&L':<12} {'P&L %'}")
    print("-" * 65)
    for p in positions:
        print(
            f"{p.symbol:<8} {float(p.qty):<8.2f} ${float(p.avg_entry_price):<11.2f} "
            f"${float(p.market_value):<13.2f} ${float(p.unrealized_pl):<11.2f} "
            f"{float(p.unrealized_plpc) * 100:.2f}%"
        )
    return positions


def get_orders(status="open"):
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    req = GetOrdersRequest(status=QueryOrderStatus(status))
    orders = trading_client.get_orders(filter=req)
    if not orders:
        print(f"No {status} orders.")
        return orders
    print(f"\n{'ID':<12} {'Symbol':<8} {'Side':<6} {'Qty':<8} {'Type':<10} {'Status'}")
    print("-" * 60)
    for o in orders:
        print(f"{str(o.id)[:8]:<12} {o.symbol:<8} {o.side.value:<6} {float(o.qty):<8.2f} {o.order_type.value:<10} {o.status.value}")
    return orders


if __name__ == "__main__":
    print("=== ACCOUNT ===")
    get_account()
    print("\n=== POSITIONS ===")
    get_positions()
    print("\n=== OPEN ORDERS ===")
    get_orders("open")
