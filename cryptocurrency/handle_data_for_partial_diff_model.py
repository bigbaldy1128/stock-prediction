# @author wangjinzhao on 2021/8/23
import glob
import sys
from datetime import *
import pandas as pd
from cryptocurrency.enums import *
from cryptocurrency.utility import download_file, get_all_symbols, get_parser, get_start_end_date_objects, \
    convert_to_date_object, get_path


def download_monthly_klines(trading_type, symbols, num_symbols, intervals, years, months, start_date, end_date, folder,
                            checksum):
    current = 0
    date_range = None

    if start_date and end_date:
        date_range = start_date + " " + end_date

    if not start_date:
        start_date = START_DATE
    else:
        start_date = convert_to_date_object(start_date)

    if not end_date:
        end_date = END_DATE
    else:
        end_date = convert_to_date_object(end_date)

    print("Found {} symbols".format(num_symbols))
    path = ""
    for symbol in symbols:
        print("[{}/{}] - start download monthly {} klines ".format(current + 1, num_symbols, symbol))
        for interval in intervals:
            for year in years:
                for month in months:
                    current_date = convert_to_date_object('{}-{}-01'.format(year, month))
                    if current_date >= start_date and current_date <= end_date:
                        path = get_path(trading_type, "klines", "monthly", symbol, interval)
                        file_name = "{}-{}-{}-{}.zip".format(symbol.upper(), interval, year, '{:02d}'.format(month))
                        download_file(path, file_name, date_range, folder)

                        if checksum == 1:
                            checksum_path = get_path(trading_type, "klines", "daily", symbol, interval)
                            checksum_file_name = "{}-{}-{}-{}.zip.CHECKSUM".format(symbol.upper(), interval, year,
                                                                                   '{:02d}'.format(month))
                            download_file(checksum_path, checksum_file_name, date_range, folder)

        current += 1
    return path


def download_daily_klines(trading_type, symbols, num_symbols, intervals, dates, start_date, end_date, folder, checksum):
    current = 0
    date_range = None

    if start_date and end_date:
        date_range = start_date + " " + end_date

    if not start_date:
        start_date = START_DATE
    else:
        start_date = convert_to_date_object(start_date)

    if not end_date:
        end_date = END_DATE
    else:
        end_date = convert_to_date_object(end_date)

    # Get valid intervals for daily
    intervals = list(set(intervals) & set(DAILY_INTERVALS))
    print("Found {} symbols".format(num_symbols))
    path = ''
    for symbol in symbols:
        print("[{}/{}] - start download daily {} klines ".format(current + 1, num_symbols, symbol))
        for interval in intervals:
            for date in dates:
                current_date = convert_to_date_object(date)
                if current_date >= start_date and current_date <= end_date:
                    path = get_path(trading_type, "klines", "daily", symbol, interval)
                    file_name = "{}-{}-{}.zip".format(symbol.upper(), interval, date)
                    download_file(path, file_name, date_range, folder)

                    if checksum == 1:
                        checksum_path = get_path(trading_type, "klines", "daily", symbol, interval)
                        checksum_file_name = "{}-{}-{}.zip.CHECKSUM".format(symbol.upper(), interval, date)
                        download_file(checksum_path, checksum_file_name, date_range, folder)

        current += 1
    return path


def handle_data():
    result = pd.DataFrame()
    df = pd.read_csv("data/btcusdt.csv", usecols=['Open time', 'Close', 'Volume'])
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
    for i in range(df.shape[0]):
        result = result.append(df.iloc[i])
        result = result.append(pd.Series(), ignore_index=True)
    result = result[['Open time', 'Close', 'Volume']]
    result.to_csv("data/btcusdt_new.csv")


def download_data():
    parser = get_parser('klines')
    args = parser.parse_args(sys.argv[1:])

    if not args.symbols:
        print("fetching all symbols from exchange")
        symbols = get_all_symbols(args.type)
        num_symbols = len(symbols)
    else:
        symbols = args.symbols
        num_symbols = len(symbols)

    if args.dates:
        dates = args.dates
    else:
        now = datetime.now()
        this_month_start = datetime(now.year, now.month, 1)
        dates = pd.date_range(start=this_month_start, end=datetime.today(), periods=MAX_DAYS).to_pydatetime().tolist()
        dates = [date.strftime("%Y-%m-%d") for date in dates]
        paths = []
        paths.append(download_monthly_klines(args.type, symbols, num_symbols, args.intervals, args.years, args.months,
                                             args.startDate, args.endDate, args.folder, args.checksum))
        paths.append(
            download_daily_klines(args.type, symbols, num_symbols, args.intervals, dates, args.startDate, args.endDate,
                                  args.folder, args.checksum))
        alldata = pd.DataFrame()

        for path in paths:
            path = path + args.startDate + '_' + args.endDate
            for file in glob.glob(path + "/*.zip"):
                print(file)
                temp = pd.read_csv(file, names=["Open time", "Open", "High", "Low", "Close", "Volume", "Close time",
                                                "Quote asset volume", "Number of trades", "Taker buy base asset volume",
                                                "Taker buy quote asset volume", "Ignore"],
                                   compression='zip')
                temp = temp.drop(['Close time',
                                  'Open',
                                  'High',
                                  'Low',
                                  'Quote asset volume',
                                  'Number of trades',
                                  'Taker buy base asset volume',
                                  'Taker buy quote asset volume',
                                  'Ignore'], axis=1)
                alldata = alldata.append(temp, ignore_index=True)
        alldata.to_csv("data/btcusdt.csv")


if __name__ == "__main__":
    # download_data()
    handle_data()
