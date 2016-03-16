attrib +r *(2).txt
attrib +r *.bat
del /q *
attrib -r *(2).txt
attrib -r *.bat

@echo off
setlocal enabledelayedexpansion
for %%i in (*) do (
	(echo "%%i" | find /i ".bat" 1>NUL) || (
		set file=%%i
		ren "!file!" "!file: (2)=!"
	)
)