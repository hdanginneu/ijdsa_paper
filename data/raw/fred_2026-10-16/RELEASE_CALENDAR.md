# Release timing of the macro inputs

Checked on 17/10/2026 from the official pages listed below. Times are US Eastern (ET) unless stated.
Forecast origin = ICE London Robusta close, **17:30 London**. Trading hours 09:00–17:30 London
(ICE contract specs). 17:30 London = **12:30 ET** in most weeks; in the 2–3 weeks a year when US and UK
daylight-saving dates differ, 17:30 London = 13:30 ET (the US shift is earlier in spring and later in
autumn, so the ICE close is never earlier than 12:30 ET).

The macro lag is counted in **coffee (London) trading days**: the value used at origin t is the one dated
6 coffee trading days before t.

| Series (FRED id) | Publisher / release | Frequency and time | Delay vs value date | Relation to the ICE close at origin t | Sources (accessed 17/10/2026) |
|---|---|---|---|---|---|
| Nominal Broad U.S. Dollar Index (DTWEXBGS) | Federal Reserve Board, H.10 Foreign Exchange Rates | Weekly, **Mondays 4:15 p.m.**, covering the previous business week; "If Monday falls on a Federal Holiday, the data will be released on the following business day" | Monday's value: 7 calendar days (5 business days); Friday's value: 3 calendar days. Holiday week: +1 business day | Release (16:15 ET) is **after** the ICE close, so a value released on Monday is first usable at Tuesday's origin. Worst case (Monday value): usable 6 US business days later; **7 when the next Monday is a federal holiday** | https://www.federalreserve.gov/releases/h10/ ; https://www.federalreserve.gov/releases/h10/default.htm ; https://fred.stlouisfed.org/series/DTWEXBGS (FRED: "Release: H.10", last updated 13/10/2026 = Tuesday after Columbus Day, next release 19/10/2026) |
| Effective Federal Funds Rate (EFFR) | Federal Reserve Bank of New York | Daily; "The New York Fed publishes the EFFR for the prior business day … at approximately 9:00 a.m." | 1 business day | Published (≈ 09:00 ET) **before** the ICE close (12:30 ET): the value for day t−1 is known at origin t | https://www.newyorkfed.org/markets/reference-rates/effr |
| WTI spot, Cushing (DCOILWTICO) | U.S. EIA, "Spot Prices" (daily data published weekly); FRED mirrors it | Weekly. Standard EIA weekly release **Wednesday 10:30 a.m.**; holiday weeks **Thursday 12:00 p.m.** (2026: 22/01, 19/02, 28/05, 10/09, 15/10, 12/11). On 17/10/2026 the EIA spot-price page showed "Release Date: 10/15/2026, Next Release Date: 10/21/2026" with daily values through 10/14 | Wednesday's value: published the following week (5 business days; 6 in a holiday week) | EIA release (10:30 or 12:00 ET) is **before** the ICE close. Worst case: usable 5 US business days later, **6 in a holiday week**. FRED posts the update later the same day (15/10/2026 at 3:12 p.m. CDT = after the ICE close) | https://www.eia.gov/dnav/pet/pet_pri_spt_s1_d.htm ; https://www.eia.gov/petroleum/supply/weekly/schedule.php ; https://fred.stlouisfed.org/series/DCOILWTICO |

Notes
- The link between the spot-price page and the Weekly Petroleum Status Report schedule is inferred from the
  matching dates (15/10 and 21/10/2026); the EIA schedule page does not say explicitly that spot-price
  tables follow the WPSR calendar [unverified].
- All three series were downloaded on 16/10/2026 as the current FRED vintage. FRED notes for DCOILWTICO state
  "All data are subject to revision"; real-time (first-release) vintages were not used [limitation].
- WTI is also a market price observable the same day by traders; the official-release timing above is the
  conservative standard.

## Conclusion 
- **EFFR**: lag 6 is far more than needed (1 business day suffices).
- **WTI (DCOILWTICO)**: lag 6 is sufficient with respect to the official EIA release (one day of slack in normal
  weeks, none in holiday weeks). If the FRED posting time is taken as the publication time instead, a value can
  be one day short in holiday weeks.
- **DTWEXBGS**: lag 6 is **just sufficient in normal weeks** (no slack) and **one coffee trading day short** for the
  Monday value of a week that is followed by a Monday US federal holiday (MLK, Presidents', Memorial, Labor,
  Columbus Day and Monday-observed holidays; roughly 5–8 values per year). A UK bank holiday in between adds a
  day of slack, because the lag is counted in London trading days.
- The configuration was **not changed** (handoff rule). The exposure is limited to a handful of DTWEXBGS values
  per year; the lag-1 robustness run already shows how results move when macro values are far more recent.
