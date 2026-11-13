# Screenshot guide for the report

Save every screenshot into `report/screenshots/` with EXACTLY these file names (PNG).
Windows: press Win + Shift + S, drag over the area, then paste into Paint (or Snipping Tool) and save.

## Before you start
1. Make sure these have been run (so the numbers in the report are final):
   `python run_stage6.py --calibrate` then `python run_stage6.py`  (about 3 minutes, uses Groq)
2. Start the app: `streamlit run app.py`. Use browser zoom 100%, a window about 1400 px wide, light theme if possible.
3. Close other tabs or crop them out. Never show `.env` or your API key.
4. Click the examples in the SIDEBAR rather than typing, so your screenshots match the report text.

## The screenshots

| File | What to capture | How |
|---|---|---|
| `01_app_overview.png` | The whole app: sidebar, answer with coloured citation, guardrail strip, Evidence panel | Click **Error code (keyword)**, capture the full browser window |
| `02_paraphrase.png` | Answer plus Evidence panel for a question with different words | Click **Different words, same meaning** (cards should show a blue "Semantic only" or green card) |
| `03_pii_masking.png` | The "Personal data masked: EMAIL" chip and the answer | Click **Personal data (PII)**, crop to the chat message and chips |
| `04_request_log.png` | The last log line proving the raw email was never stored (query shows `[EMAIL]`) | In PowerShell run `Get-Content logs\requests.jsonl -Tail 1` with a wide window; crop so the `"query"` part is readable |
| `05_out_of_scope.png` | A refusal with no sources | Click **Out of scope** (answer pill "Out of scope", Evidence panel explains why) |
| `06_near_domain_trap.png` | The Gmail question being answered "not in the knowledge base" | Click **Near-domain trap** |
| `07_injection_blocked.png` | The "Blocked" pill | Click **Prompt injection** |
| `08_pytest.png` | All tests passing | Terminal: `python -m pytest -q`, capture the last lines (dots and "passed") |
| `09_project_structure.png` | The folder layout | VS Code Explorer with `config`, `data`, `src/helpdesk`, `tests`, `docs`, `results` expanded |
| `10_stage6_summary.png` (optional) | The summary from the end-to-end run | Terminal: the "Status counts" and following lines of `python run_stage6.py` |

## Then build the report
1. `pip install -r requirements.txt` (adds reportlab and pillow)
2. Edit `report/report_meta.yaml` (your name, BITS id, optional group note)
3. `python build_report.py`
4. Open `report/Conversational_AI_Assignment1_Report.pdf`. Red boxes mean a screenshot is missing; the script also lists them.
   Use `python build_report.py --strict` to make it fail until every required screenshot exists.
