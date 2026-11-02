# Chronos-Bolt (small): release date and training corpus

Checked 22/10/2026. Only what the cited pages state is reported as fact; everything else is marked [unverified].

| Item | Finding | Source |
|---|---|---|
| Model used | `amazon/chronos-bolt-small` (zero-shot; context 512 closes; median as point forecast) | `scripts/run_chronos.py` |
| Release date | GitHub News: **"26 Nov 2024: Chronos-Bolt models released"**. The Hugging Face model card gives no original release date (it mentions an "Update Feb 14, 2025" about SageMaker JumpStart). | https://github.com/amazon-science/chronos-forecasting ; https://huggingface.co/amazon/chronos-bolt-small |
| Training data of Chronos-Bolt | Model card: "trained on nearly 100 billion time series observations". **No list of the Bolt training datasets is given** on the model card or the GitHub page. **[unverified]** whether Bolt used exactly the original Chronos corpus. | same pages |
| Original Chronos training corpus | Ansari et al. (2024), *Chronos: Learning the Language of Time Series*, TMLR, arXiv:2403.07815, Table 3 / Appendix B: 28 training datasets (13 pretraining-only + 15 in-domain Benchmark I), ≈890K univariate series, plus synthetic data. Domains: energy, transport, nature/weather, web, and M4 ("various"). | https://arxiv.org/html/2403.07815 |
| Finance / commodity data in the training set | Per Table 3, the finance/economics datasets of the paper (CIF 2016, Exchange Rate 1990–2016, FRED-MD, NN5) are **zero-shot evaluation only, not training**. No commodity-price dataset is listed among the training datasets. | same |
| M4 (Daily) | In the training set (domain "various", 4,227 series, mean length 2,371). The paper says M4 as a whole covers "micro-/macroeconomics, demographic, industry, and finance" but does not say whether the daily subset contains financial series, nor its years. The M4 competition was run in 2018, so its series cannot extend past early 2018 **[unverified: exact end dates not stated in the paper]**. | same |
| Latest dated training data found | Wiki Daily (100k) 2007–2022; Mexico City Bikes 2010–2022; Pedestrian Counts 2009–2020; WeatherBench 1979–2018. | same |

## Implication for the evaluation blocks (factual, no interpretation)
- B1–B4 origins: 2020-12 to 2024-11. Training data dated up to 2022 exist in the corpus (none of them coffee or
  commodity prices per Table 3); whether any ICE coffee series entered the Bolt corpus cannot be verified **[unverified]**.
- B5 origins start 2024-11-19, one week before the Chronos-Bolt release (26/11/2024); B6 starts 2026-01-28.
  Data after the release date cannot be in the training corpus. B5–B6 are therefore the cleanest zero-shot
  test available; `tables/v3_chronos_b56.csv` reports Chronos and SHRINK-Chronos on B5–B6 only.
