def run_app():
    import streamlit.web.cli as stcli
    import sys
    from pathlib import Path

    file_path = Path(__file__).parent / "app.py"
    sys.argv = ["streamlit", "run", str(file_path)]
    stcli.main()