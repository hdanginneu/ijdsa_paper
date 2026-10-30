# Contract roll notes

Checked 22/10/2026. Only what the cited pages state is reported as fact.

## ICE Robusta Coffee Futures (ICE Futures Europe) — https://www.ice.com/products/37089079/Robusta-Coffee-Futures/specs
- Delivery months: "January, March, May, July, September, November, such that ten delivery months are available for trading".
- First Notice Day: "Fourth business day preceding the first business day of the delivery month."
- Last Trading Day: "Fourth business day preceding the last business day of the delivery month at 12:30."
- Last Notice Day: "Fourth business day preceding the last business day of the delivery month."
- Contract size: "Ten Tonnes"; price quotation "$ per metric tonne". Trading hours 09:00–17:30 London.

## ICE Coffee "C" Futures (ICE Futures U.S., Arabica) — https://www.ice.com/products/15/Coffee-C-Futures/specs
- Contract series: "March, May, July, September, December".
- First Notice Day: "Seven business days prior to first business day of delivery month."
- Last Trading Day: "One business day prior to last notice day".
- Contract size: "37,500 pounds"; quotation in cents per pound. (Converted to USD/t in this project;
  a 10-t lot for Arabica in D-51 is a comparison convention only, not the ICE contract size.)

## Investing.com continuous series
- How Investing.com builds its continuous "London Coffee" / "US Coffee C" history (which contract on which
  date, and whether prices are back-adjusted) could not be verified: the Investing.com support pages were
  not reachable from this machine on 22/10/2026 (connection closed). **[unverified]**
- No adjustment factors or contract identifiers are present in the files.

## Anchors used by scripts/roll_calendar.py (assumptions, stated explicitly)
- Business days are approximated by the trading days present in each file (exchange holiday calendars
  are not modelled separately).
- **Primary anchor (D-49): First Notice Day**, computed as the 4th (Robusta) / 7th (Arabica) trading day
  before the first trading day of the delivery month in the file.
- **Supplementary anchor (added, not in D-49): Last Trading Day**, computed as the 4th (Robusta) trading day
  before the last trading day of the delivery month, and for Arabica one trading day before the Last
  Notice Day approximated as the 7th trading day before the last trading day of the delivery month
  ([unverified]: the Coffee "C" last-notice rule is not quoted above). Reason: Robusta LTD falls about one month after
  FND, and the Investing.com roll date is unknown ([unverified]), so a roll near expiry would be outside the
  FND windows. Results for the LTD anchor are reported as separate versions (`flagged_ltd`, `window_ltd`).
- Window: ±10 trading days around the anchor. Flag: the day with the largest overnight gap
  |log open_t − log close_{t−1}| in the window if it exceeds 3 × the full-sample median gap; otherwise none.

## Data limitation for the gap measure
- The Robusta file has 362 rows with open = high = low = close (mostly early years of the earlier export). On
  those days the "overnight gap" equals the whole close-to-close move, so the gap is not a clean overnight
  measure there. The rows are kept as in the file (D-17); the full-sample median gap is computed over all rows.
