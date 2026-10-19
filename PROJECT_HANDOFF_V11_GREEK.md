# EuroLeague Team Markets v1.1 — Πλήρες handoff για νέα συνομιλία

**Κατάσταση (20/10/2026): RESEARCH/SHADOW — ΟΧΙ ΕΓΚΕΚΡΙΜΕΝΕΣ ΕΜΠΟΡΙΚΕΣ ΑΠΟΔΟΣΕΙΣ.**

## 0. Τι ακριβώς ζητά ο χρήστης
Ένα εργαλείο για pre-event basketball traders: όλες οι αγορές ομάδων (PTS, 2PM/2PA, 3PM/3PA, FTM/FTA, OREB/DREB/REB, AST, TO, STL, BLK), team/combined game totals, 3-way most statistic με πραγματική ισοπαλία και handicaps. Πάντα δίπλα-δίπλα **fair probabilities/odds** και **bookmaker decimal odds** με άθροισμα implied probabilities 109% για δυαδικές και 115% για 3-way. Ordered trader board και CSV export. Η ασφάλεια και η σωστή στατιστική μεθοδολογία προηγούνται της ταχύτητας. Ειδικά ο χρήστης απαιτεί **live predictions αποκλειστικά από την ΤΡΕΧΟΥΣΑ σεζόν της ομάδας, του αντιπάλου της και της λίγκας**· οι προηγούμενες χρονιές χρησιμοποιούνται μόνο για εκμάθηση καθολικών coefficients/priors και error model. **ΠΟΤΕ περσινοί team rates στα live inputs**.

## 1. Δεδομένα και settling
Ιστορικές σεζόν 2023, 2024, 2025 και τρέχουσα 2026. Upload `team_games (3).csv`, `player_games (3).csv` ήταν οι πιο πρόσφατες εκδόσεις, έως 16/10/2026. 1.051 παιχνίδια συνολικά, 60 τρέχοντα· 58 κανονικής διάρκειας μετά το OT exclusion. Οι επίσημες team market τιμές REB/TO **ΠΕΡΙΛΑΜΒΑΝΟΥΝ TEAM-ONLY**. `MARKET_REB=MARKET_OREB+MARKET_DREB`, `STL<=OPP_PLAYER_CREDITED_TO`. Player-only turnovers ΔΕΝ ισούνται με όλα τα official team TO.

Τα ομαδικά μοντέλα και Monte Carlo εκτιμούν 40λεπτη κανονική διάρκεια. **Εξωτερικές bookmaker αγορές συχνά περιλαμβάνουν παράταση**, επομένως η σύγκριση δεν είναι πλήρως συγκρίσιμη χωρίς OT module.

## 2. Αρχική αρχιτεκτονική και προβλήματα
Η v0.6 χρησιμοποίησε δομικό shot-allocation model με κοινές κατοχές, αλλά επέλεγε `league` για 3P%, OREB, STL και near-zero coefficients στα DREB, οδηγώντας σε υπερβολική σύγκλιση. Το v0.8 αυξήθηκε η διαφοροποίηση pace/3PA, αλλά είχε επιρροή περσινών team stats στο live forecast. v0.9 shrinkage-only ήταν ξεχωριστό **audit** και ΔΕΝ άλλαζε τις τιμές. v1.0 διόρθωσε το live current-only input, αλλά επέλεγε rates ενώ ο trader χρειάζεται σωστά τελικά counts και δεν περιείχε επαρκές home court correction. Η παλιά MC έδινε ~99% coverage για 90% intervals στα REB λόγω υπερβολικής variance. Μην επαναφέρεις αυτά τα λάθη.

## 3. Πραγματική υλοποίηση v1.1
**Κεντρικός κώδικας** `core/matchup_v11.py`, `core/margin_v11.py`, `core/rotation_audit_v11.py`.

1. Same-season cumulative team offensive counts και opponent-allowed counts (μόνο παιχνίδια πριν την ημερομηνία as-of, ίδια ημερολογιακή μέρα embargoed). Same-season league baseline. Καμία περσινή ομάδα ως live prior.
2. Historical grid per market: pseudo-games `k in {0,2,4,8,16}` και opponent coefficient `β in {0,.25,.5,.75,1}`. Επιλογή με count-loss στο 2024, *μη επιλεγμένο* 2025 historical test, και 2026 watch (ήδη εξετασμένο). Τύπος:
   `PRED_k = own_current_shrunk_to_current_league + β_k*(opponent_allowed_current_shrunk_to_current_league − current_league)`.
   Για percentages, σχετική pseudo-exposure με attempts· **όχι** νέα τυχαία weights ανά συγκεκριμένο matchup.
3. Κοινό pregame pace/shot allocation με variance-weighted προβολή σε ακριβείς 2 σχέσεις:
   `PACE ≈ 2PA + 3PA + 0.44*FTA + official_TO − official_OREB` ανά ομάδα. Home/away κοινό PACE. Εκτιμημένες covariance-scale SD από historical 2024 residuals. Θεωρητικό box-score possession estimator, όχι ακριβές PBP.
4. Made shots / points: `2PM=2PA*2P%`, `3PM=3PA*3P%`, `FTM=FTA*FT%`, `PTS=2*2PM+3*3PM+FTM`. Home court applied to **makes** (2PM,3PM,FTM) using symmetric effect fitted on 2024, tested 2025. β_home makes: see report `home_adjustment_from_2024_made_baskets`.
4a. **ΑΝΕΞΑΡΤΗΤΟ MARGIN / TEAM STRENGTH MODEL**: από παλιές σεζόν μαθαίνουμε home-court baseline (2023), `net_prior_games=2`, `net_coefficient=0.75` (2024 CV). Live team net performance και opponent net performance είναι **μόνο φετινά**. Συνδυάζει την independent home-spread estimate 50% με τον joint structural home margin 50%, βάρος επιλεγμένο **το 2024** και ελεγμένο **το 2025**. Το margin μετατρέπεται σε πραγματικά 2PM/3PM/FTM adjustments μέσω variance-optimal learned weights (FG2M 34.9%, FG3M 49.5%, FTM 15.6% των επιπλέον πόντων), με ταυτότητα `PTS=2*2PM+3*3PM+FTM` χωρίς αυθαίρετο score bonus. Σκοπός να ΜΗΝ προβλέπει σχεδόν κάθε παιχνίδι pick'em όταν οι φετινές ομάδες έχουν διαφορετική καθαρή διαφορά πόντων. Το ανεξάρτητο 2025 RMSE στο spread βελτιώνεται ελάχιστα από ~11.953 σε ~11.943, και στο παρακολουθημένο 2026 από ~11.450 σε ~11.302. Το μέγεθος βελτίωσης παραμένει μικρό, όχι bookmaker proof.
5. OREB official market gets current own+opponent model; DREB gets own+opp current estimates plus checked bound from opponent miss opportunities (missed FG + 0.5 missed FT - opponent OREB), with 2-event unknown dead-ball margin. **Exact reboundable FTs require PBP and this is not solved.** Official `REB=OREB+DREB` always.
6. Steals capped using opponent *player-credited* turnovers; blocks capped on missed opponent 2PA; AST <= successful team field goals.
7. Uncertainty: `matched_holdout_2025_residuals` paired game bootstrap into integer team box scores, retaining paired errors and mathematically enforced points/REB; tested chronological 2026 watch via `audit_distribution_v11.py`.
8. UI `app_trader_panel_v11.py`: visible current offense vs opponent allowed audit, all team means, 109%/115% fair/book price-preview, sorted team + game + 3-way + handicap, custom external odds sanity input, player rotation audit, **HOLD mandatory** (not production calibration).
9. `app_euroleague.py` updated to load new v1.1 board and no longer surface legacy v0.7 trader board.
10. Github Actions `.github/workflows/euroleague-refresh.yml` trains after daily refresh, audits and verifies `shadow_v11` JSON and force-stages only JSON outputs. Does not require new bootstrap.

## 4. Πραγματικά test results
`python -m pytest -q` => **47 passed** (ο έλεγχος πραγματοποιήθηκε σε περιβάλλον Python χωρίς εγκατεστημένο Streamlit, άρα το Streamlit GUI ΔΕΝ ελέγχθηκε σε εκτελούμενο browser).

**Out-of-year 2025 test (690 team sides, same-season pregame features):**
- PTS RMSE 10.092, FG3A 4.597, FG3M 2.911, FTA 6.144, OREB 3.671, DREB 4.080, REB 5.372, TO 3.459, AST 3.988, STL 2.477.
- FG3A independent offense-only baseline 4.941 versus opponent-adjusted + structural 4.597, improvement approximately 7.0%.

**2026 watch (38 distinct games, 76 team sides; NOT untouched):**
- PTS 10.559, FG3A 5.112, FG3M 3.070, FTA 6.895, OREB 3.637, DREB 4.016, REB 5.327, TO 3.189, STL 2.506.
- Prior v1.0 on same 2026 watch: PTS 10.671, FG3A 5.253, REB 5.431, TO 3.077, STL 2.544. **Turnovers are slightly worse in v1.1**, so do NOT claim every market improved.

**2026 watch empirical paired-bootstrap 90% interval coverage, n=76 team sides:**
PTS 90.8%, FG3A 85.5%, OREB 92.1%, DREB 90.8%, REB 89.5%, FTA 86.8%, AST 94.7%, TO 94.7%, STL 89.5%, BLK 97.4%. **Not enough for bookmaker odds approval.**

**380 theoretical directional matchups across 20 teams, as of 21/10/2026:**
PACE 71.07–78.65; total points 162.91–182.88; home spread range −5.50 έως +12.80; team home 3PA 20.21–37.02; home REB 33.39–38.05. 108 of 380 scenarios flagged as needing more current-season games (teams with 5 regulation games instead of 6) apart from blanket distribution HOLD. These matchups are a stress-test, not the actual EuroLeague fixture list.

## 5. Concrete matchup outputs / sanity
Using **the dataset through 16/10 and as-of 21/10/2026**, including empirically learned 2024 home court effect:

| Matchup | Pace | Home PTS | Away PTS | Total | Home margin | Home 3PA | Home REB |
|---|---:|---:|---:|---:|---:|---:|---:|
| PAN–BAS | 75.95 | 92.17 | 79.37 | 171.54 | +12.80 | 24.60 | 35.80 |
| HTA–VIR | 76.79 | 90.50 | 85.77 | 176.27 | +4.72 | 27.95 | 36.09 |
| PRS–VIR | 77.51 | 87.93 | 87.64 | 175.57 | +0.29 | 34.65 | 36.89 |

Hapoel–Virtus **published betting comparison (as of 20/10)**: BetMGM full-game incl. OT Hapoel −8.5, total 173.5, Hapoel moneyline 1.30. The model gives regulation +4.72 and total 176.27. **This is a meaningful external difference, especially margin; STOP/HOLD, verify availability, venue, roster, current injuries and book settlement; do not tune the model to bookmaker.** Source: https://www.betmgm.ca/en/sports/events/hapoel-tel-aviv-isr-virtus-bologna-ita-20014863 (market differs by OT inclusion).

PAN–BAS external full-time market: **ΠΑΟ −15.5**, **total 170.5**, vs regulation model **ΠΑΟ −12.80**, **total 171.54**; does not imply the market must be copied. Source: https://www.betmgm.ca/en/sports/basketball-7/betting/europe-7/euroleague-men-870 . Full-time includes OT, different settlement.

ΠΑΟ–Μπασκόνια opponent audit: ΠΑΟ own 3PA ~26.67 this season, Baskonia allowed ~22.5, even excluding anomalous Virtus/Dubai games Baskonia allowed less than opponents' other games. Opponent-adjusted 24.6 3PA reasonable; independent count baseline earlier ~24.18. BAS total rebounds ~34.84 supported by own current ~35.17 vs PAN allowed ~33.50, but exact DREB opportunities still lack PBP. **Do not force a line to sound intuitive.**

## 6. Known unsolved issues / ABSOLUTE STOP FOR PRODUCTION
- No confirmed projected lineups, injuries, minute limits, player minutes forecast, coach-state regime segmentation or travel adjustments. `core/rotation_audit_v11.py` only produces descriptive current-season player/position-category flags; it does NOT adjust the mean.
- Only 38 prospective-ish games in frequently inspected 2026 watch; distribution not independently validated, no strict proper scoring-rule validation. Model variance calibration not approved.
- Boxscore reboundability proxy for FT and team-only possessions is not a genuine possession PBP model. Do not claim precise rebound odds.
- Opponent sample small, `k`/β results subject to sampling uncertainty. Home/away margin discrimination remains weak versus independent markets for some matches (e.g. HTA–VIR).
- Bookmaker markets may include OT while the engine excludes it. Need clear matching settlement and OT module before live use.
- 380 stress predictions do not establish that *every* line is correct. A model cannot guarantee no odd line; publish HOLD rather than false confidence.
- Browser integration not executed in the local container because Streamlit is not installed. All Python syntax and unit tests passed; GitHub/Streamlit deployment unverified.

## 7. GitHub deployment (ONLY AFTER USER APPROVES RESEARCH/SHADOW)
1. Copy relative patch files preserving root folders; especially `.github/workflows/euroleague-refresh.yml`, `core/matchup_v11.py`, `core/margin_v11.py`, `core/rotation_audit_v11.py`, `app_trader_panel_v11.py`, modified `app_euroleague.py`, `train_matchup_v11.py`, `audit_distribution_v11.py`, `verify_matchup_v11.py` and `tests/test_matchup_v11.py`.
2. Commit to `main`; run Actions→EuroLeague Data Refresh→`daily` (not bootstrap).
3. Verify 3 new steps green: fit v1.1, audit v1.1 distributions, verify v1.1 gates. Ensure `data/euroleague/shadow_v11/matchup_v11_report.json` and `distribution_v11_report.json` are committed by runner.
4. Streamlit: new `Trader Board v1.1 AUDITED` panel. **Release HOLD** is expected and REQUIRED, not a failure.
5. No changes to WNBA application.

## 8. Bibliography / independent principles
- Oliver, D. (2004), *Basketball on Paper*, Four Factors and possession accounting.
- Kubatko, J., Oliver, D., Pelton, K., Rosenbaum, D. (2007), *A Starting Point for Analyzing Basketball Statistics*, Journal of Quantitative Analysis in Sports.
- Migliorati, M., Manisera, M., Zuccolotto, P. (2023), Four Factors, model-based recursive partitioning, https://pmc.ncbi.nlm.nih.gov/articles/PMC9251030/.
- Tzai et al. (2023), *Predictions of European Basketball Match Results with ML*, https://journals.sagepub.com/doi/10.3233/JSA-220639.
- Gneiting, T., Raftery, A. (2007), strictly proper scoring rules / calibration of probabilistic forecasts, https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf.

**Important**: literature supports principles, not the exact tested hyperparameters of v1.1. Those are empirically selected by the specified year split and must be retested with genuinely new rounds.

## 9. Exact prompt for next chat
> Συνεχίζουμε το EuroLeague team-market project από την **v1.1**. Διάβασε πρώτα το `PROJECT_HANDOFF_V11_GREEK.md`. ΜΗ χρησιμοποιήσεις περσινές τιμές ομάδων σε live predictions: 2023–25 ONLY train coefficients and error distributions, current 2026 ONLY offense/opponent allowed/league inputs. Τα official MARKET_REB/TO περιλαμβάνουν team-only. Έχουμε ιστορικό 2025 test και 2026 watch· 47 Python tests passed αλλά production quote gate είναι HOLD, ειδικά στα rebounds, FT, steals, margin και OT. Θέλω ανεξάρτητο χρονικό calibration και επαληθευμένες απουσίες/ρόστερ/παικτικά roles πριν οποιαδήποτε δημοσίευση τιμών. Μη συγχέεις σωστή γκανιότα 109%/115% με σωστή forecast distribution. Εξήγησε γιατί κάθε line και γιατί οποιαδήποτε απόκλιση από τις εξωτερικές εταιρείες, χωρίς να κουρδίζεις στο market. Δώσε συγκεκριμένα validation tables, όχι αυθαίρετες αλλαγές.
