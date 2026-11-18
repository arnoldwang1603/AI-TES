# Temperature database, 2026-11-15 download

Copy of the data group's Drive folder `Model development database/Temperature model_ most-recent-data (auto-updated)` as downloaded on 2026-11-15 (latest file time on Drive 2026-11-12 15:11). Same flat layout as `../data/`: `Latest Database (Use this for training)/` (312 files) and `Test_70_cases/` (70 files). `MANIFEST_md5.csv` lists every file with its md5 and its path in the download.

One change from the download: `Case 8 (linear discharging)/ Discharge 14- Erick WKCp.csv` (leading space in the Drive name) is stored as `Discharge 14- Erick WKCp.csv`, the name it had in `../data/`. Its content is byte-identical to the old file.

Compared with `../data/` (the Round 10 data):

- `Test_70_cases/`: all 70 files byte-identical.
- Training: 233 files identical, 146 re-exported (Time and Input unchanged in every file), 2 added (`Case 1 (linear charging)/2nd Run Case new 18 (1 second s.csv`, `Case 4 (Linear Charging+Discharging)/2nd Run Case new 10.csv`), 1 removed (`Case 1 (linear charging)/Case new 20.csv`).
- T_avg is a line average in every training file (0 inner-weighted).
- 18 training files still have T_inner from the HX face (T_inner = Input); every other file has the concrete inner surface. The split `src/models/GRU/splits/newdata_1115_s42.json` masks T_inner on the 16 of them it keeps (Case 10) and drops the two `2nd Run` files.

Audit: calit2-thermal-wiki `outputs/data-audit-2026-11-15/report_zh.md`.
