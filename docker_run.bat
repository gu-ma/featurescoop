@echo off

@REM Set a variable with additional command line arguments
:loop
set args=%args% %1
shift
if not "%~1"=="" goto loop

@REM Run docker command
SET mypath=%~dp0
docker run %args% -p 5003:5003 -it -v %mypath%:/scratch featurescoop:latest bash -c "(cd /scratch/featurescoop && python api.py)"