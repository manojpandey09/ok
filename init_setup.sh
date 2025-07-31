@echo off
echo [%date% %time%]: START

echo [%date% %time%]: Creating env with Python 3.8 version
call conda create --prefix ./env python=3.8 -y

echo [%date% %time%]: Activating the environment
call conda activate ./env

echo [%date% %time%]: Installing the dev requirements
pip install -r requirements.txt

echo [%date% %time%]: END
pause
