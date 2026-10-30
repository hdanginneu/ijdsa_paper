# Price data (to 30 September 2026)

- `london_robusta_daily_2008-01-14_2026-09-30.csv`: ICE London Robusta futures, Investing.com continuous series, daily close, open, high, low and volume (4,773 rows). Closes up to 27 Jan 2026 come from an earlier export of the same series; from 2 Jan 2026 the file uses values fetched from Investing.com's historical-data page on 16 Oct 2026 (the two sources agree on all overlapping closes).
- `us_coffee_c_daily_2008-01-14_2026-09-30.csv`: ICE US Coffee "C" futures (Arabica), Investing.com continuous series, fetched 17 Oct 2026; US cents/lb, converted to USD/t with 22.0462262. Used only in the Arabica robustness run.
- `london_2026_from_api.csv`: the 2026 London Robusta rows as fetched.
- `ROLL_NOTES.md`: contract-roll rules from the ICE specification pages and what is and is not known about Investing.com's roll date.

Use of the Investing.com data remains subject to the provider's terms.

## Known issues
1. 28 Jan 2026: an earlier export shows a close of 4,148 with empty volume; the final close is 3,984. The earlier value is probably an intraday price taken on the download day. The final value is used.
2. 21-24 Sep 2026: open = high = low = close and volume 0 (possible filler rows). The main run keeps the closes and sets volume to missing; a sensitivity run drops these rows.
3. 362 London rows and 90 Arabica rows have open = high = low = close (mostly early years); they are kept.
