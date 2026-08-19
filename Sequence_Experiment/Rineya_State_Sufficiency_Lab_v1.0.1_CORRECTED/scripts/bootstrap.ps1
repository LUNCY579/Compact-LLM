$ErrorActionPreference = "Stop"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[train,dev]"
python -m rineya_state_lab doctor
