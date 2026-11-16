NOTEPROOF — AI-FEST 2026 (AI STREAM)
====================================

START THE APP (WINDOWS)
1. Extract this folder to your Desktop (if using the ZIP).
2. Install Python if it is not already installed.
3. In this folder, copy .env.example and rename the copy to .env.
4. Open .env and add your own Gemini API key. Keep .env private; never submit it.
5. Double-click RUN_NOTEPROOF.bat, or run these commands in PowerShell:
     python -m pip install -r requirements.txt
     python -m streamlit run app.py
6. Open http://localhost:8501 in your browser.

IMPORTANT
- localhost works only while the app is running on this computer.
- Do not share your API key or include your .env file in any submission.
- The 30-claim CSV is a proposed manually labelled test set. It has not yet been run
  through the app, so it is not an accuracy result.
- The five saved historical outputs are 4 SUPPORTED, 0 CONTRADICTED and
  1 UNVERIFIABLE. This distribution is not an accuracy score.
- The app source and requirements are included as available in this package.
  If you have made newer edits to your own local app.py, keep your latest version.

FOLDER CONTENTS
app.py                         Streamlit application
requirements.txt               Python dependencies
.env.example                   Safe template; add your own key in a private .env
RUN_NOTEPROOF.bat              Windows launch helper
.gitignore                     Helps avoid sharing secrets/cache files
data/                          Lecture PDF, saved JSON, evaluation CSVs, test set
submission/                    Updated presentation, evaluation report, hackathon guide

CREDITS
Python, Streamlit, PyMuPDF, Google Gemini API, pandas, scikit-learn, python-dotenv.
Lecture source: smoothing.nlp.3.pdf.
