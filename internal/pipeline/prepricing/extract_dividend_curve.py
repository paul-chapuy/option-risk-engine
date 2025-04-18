from internal.infra.api.yahoo_finance import YahooClient
import pandas as pd


def clean_option_chains(raw_chains: dict) -> pd.DataFrame:
    """
    Cleans option chain data:
    - Flattens all expiries into a single DataFrame
    - Adds expiry and option type columns
    - Keeps only relevant columns
    - Filters out rows with:
        - NaN volume
        - lastTradeDate not on the most recent trading day
    """
    all_options = []

    for expiry, chain in raw_chains.items():
        for opt_type in ["calls", "puts"]:
            df = chain[opt_type].copy()

            df = df[df["volume"].notna()]

            if not df.empty:
                most_recent_day = df["lastTradeDate"].max().normalize()
                df = df[df["lastTradeDate"].dt.normalize() == most_recent_day]

                df = df[
                    ["lastPrice", "strike", "lastTradeDate", "volume", "bid", "ask"]
                ].copy()
                df.rename(
                    columns={
                        "lastPrice": "last_price",
                        "lastTradeDate": "last_trade_date",
                    },
                    inplace=True,
                )

                df["expiry"] = expiry
                df["type"] = opt_type[:-1]
                df["mid_price"] = df[["bid", "ask"]].mean(axis=1)

                all_options.append(df)

    if not all_options:
        return pd.DataFrame(
            columns=[
                "expiry",
                "type",
                "strike",
                "last_price",
                "last_trade_date",
                "volume",
            ]
        )

    return pd.concat(all_options, ignore_index=True)


def main():
    yahoo_client = YahooClient("SPY")
    last_price = yahoo_client.get_last_price()
    print(f"last price: {last_price}")
    option_chains = yahoo_client.get_all_option_chains()
    cleaned_option_chains = clean_option_chains(option_chains)
    cleaned_option_chains.to_csv("internal/pipeline/prepricing/test.csv")


if __name__ == "__main__":
    main()
