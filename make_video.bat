@echo off
cd /d E:\stsc_frames

set FPS=240
set OUT=E:\stsc_frames\replay.mp4

echo Building video from frames in E:\stsc_frames ...
echo FPS: %FPS%
echo Output: %OUT%
echo.

ffmpeg -y -framerate %FPS% -i pixel_%%06d.png ^
    -c:v libx264 -preset fast -crf 18 ^
    -pix_fmt yuv420p ^
    -vf "scale=1920:1080:flags=lanczos" ^
    %OUT%

echo.
if %ERRORLEVEL%==0 (
    echo Done: %OUT%
) else (
    echo FAILED
)
pause
