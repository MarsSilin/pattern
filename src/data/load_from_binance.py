import logging

import pandas as pd
import requests

logger = logging.getLogger("file_logger")


def load_stocks(
    symbol: str,
    start: str,
    end: str,
    interval: str,
):
    """
    Возвращает Pandas.DataFrame, содержащий данные торгов
    :param symbol: Тикер ценной бумаги
    :param start: Дата вида ГГГГ-ММ-ДД
    :param end: Дата вида ГГГГ-ММ-ДД
    :param interval:
    :return Pandas.DataFrame:
        данные торгов в структуре: begin,open,close,high,low,value
    """
    start_time = pd.Timestamp(start)
    end_time = pd.Timestamp(end)
    start_time = int(start_time.timestamp() * 1000)  # перевод в миллисекунды
    end_time = int(end_time.timestamp() * 1000)  # перевод в миллисекунды

    df_res = pd.DataFrame(
        columns=["begin", "open", "high", "low", "close", "value"]
    )
    df_res["begin"][0] = 0
    print(df_res)

    try:
        while (
            df_res.empty
            or int(df_res["begin"].iloc[-1].timestamp() * 1000) < end_time
        ):
            url = (
                f"https://api.binance.com/api/v3/klines?symbol={symbol}"
                f"&interval={interval}&startTime={start_time}"
                f"&endTime={end_time}&limit=1000"
            )
            response = requests.get(url)
            data = response.json()

            df = pd.DataFrame(
                data,
                columns=[
                    "begin",
                    "open",
                    "high",
                    "low",
                    "close",
                    "value",
                    "Close time",
                    "Quote asset volume",
                    "Number of trades",
                    "Taker buy base asset volume",
                    "Taker buy quote asset volume",
                    "Ignore",
                ],
            )

            df = df[["begin", "open", "high", "low", "close", "value"]]
            df["begin"] = pd.to_datetime(df["begin"], unit="ms")

            df_res = pd.concat([df_res, df], ignore_index=True)
            last = int(df["begin"].iloc[-1].timestamp() * 1000)
            prelast = int(df["begin"].iloc[-2].timestamp() * 1000)
            start_time = last + (last - prelast)

        logger.debug("Получили данные биржи. Строк: " + str(len(data)))
    except Exception as ex:
        logger.exception(f"Ошибка: {ex}")

    return df_res
