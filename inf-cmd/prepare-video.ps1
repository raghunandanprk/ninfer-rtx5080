# Prepare a video for video.ps1. A local file, or any URL yt-dlp accepts (YouTube included).
#
# The engine samples 2 fps (4..768 frames) and rejects a video over 600 s, over 256 MiB, or whose
# sampled frames exceed 128 Mi decoded pixels (src/media/decode/decode.h); no server flag changes
# these. This drops the audio, caps the frame rate at 4 fps, speeds a video over 590 s up to 590 s,
# and scales it to the largest size the engine accepts. The engine never samples more than 768
# frames and downsamples every video to its token budget anyway, so nothing the model would see is
# lost, except that times the model reports in a sped-up video are compressed by the speed-up.
# A local MP4 that already fits is returned unchanged.
#
# For a URL it also saves the English subtitles (manual, else automatic) as a plain-text
# transcript: Qwen3.8 cannot hear, so paste it into the prompt for anything said aloud.
[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true, Position = 0)][string]$Source,
    [string]$OutDir = (Join-Path $env:TEMP "ninfer-video")
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$Transcript = $null

if ($Source -match '^https?://') {
    # Video only, at most 720p: fitting scales long videos well below that anyway. Only the
    # original English tracks are requested: "en.*" also matches YouTube's machine-translated
    # tracks, whose extra requests draw HTTP 429. --ignore-errors keeps a subtitle failure from
    # aborting the video, so success is judged by the downloaded file.
    $Video = yt-dlp --no-playlist --quiet --no-warnings --ignore-errors -f "bv*[height<=720]/b[height<=720]/bv*/b" `
        --write-subs --write-auto-subs --sub-langs "en,en-orig" --sub-format vtt `
        -o (Join-Path $OutDir "%(id)s.%(ext)s") --print "after_move:filepath" $Source
    $Video = @($Video | Where-Object { $_ -and (Test-Path -LiteralPath $_) })[-1]
    if (-not $Video) { throw "yt-dlp could not download $Source" }
    $Id = [IO.Path]::GetFileNameWithoutExtension($Video)
    $Subtitles = Get-ChildItem -LiteralPath $OutDir -Filter "$Id.*.vtt" | Sort-Object { $_.Name -notmatch '\.en\.vtt$' } | Select-Object -First 1
    if ($Subtitles) {
        $Lines = [Collections.Generic.List[string]]::new()
        foreach ($Line in Get-Content -LiteralPath $Subtitles.FullName -Encoding UTF8) {
            if (-not $Line -or $Line -match '^(WEBVTT|Kind:|Language:|NOTE)' -or $Line -match '-->' -or $Line -match '^\d+$') { continue }
            $Text = ($Line -replace '<[^>]+>', '').Trim()
            # Automatic captions repeat each line as the next one scrolls in.
            if ($Text -and ($Lines.Count -eq 0 -or $Lines[$Lines.Count - 1] -ne $Text)) { $Lines.Add($Text) }
        }
        $Transcript = Join-Path $OutDir "$Id.transcript.txt"
        [IO.File]::WriteAllText($Transcript, ($Lines -join "`n"), [Text.UTF8Encoding]::new($false))
    }
} else {
    $Video = (Resolve-Path -LiteralPath $Source).Path
}

$Probe = ffprobe -v error -select_streams v:0 -show_entries "stream=width,height,nb_frames,avg_frame_rate:format=duration" -of json $Video | ConvertFrom-Json
$Stream = $Probe.streams[0]
$Rate = $Stream.avg_frame_rate.Split("/")
$Fps = [double]$Rate[0] / [double]$Rate[1]
$Duration = [double]$Probe.format.duration
$Frames = 0
if (-not [int]::TryParse("$($Stream.nb_frames)", [ref]$Frames) -or $Frames -le 0) {
    $Frames = [int][math]::Round($Duration * $Fps)
}
function Get-Sampled([int]$Count, [double]$Rate) {
    [math]::Min([math]::Min([math]::Max([int][math]::Floor($Count / $Rate * 2), 4), 768), $Count)
}
$Limit = [double]128MB
$Sampled = Get-Sampled $Frames $Fps
$Fits = $Duration -le 600 -and (Get-Item -LiteralPath $Video).Length -le 256MB -and
    [double]$Sampled * $Stream.width * $Stream.height -le $Limit

if ($Fits -and [IO.Path]::GetExtension($Video) -eq ".mp4" -and $Source -notmatch '^https?://') {
    $Fitted = $Video
    $Note = "original file"
} else {
    $Speed = [math]::Max(1.0, $Duration / 590)
    $OutFps = [math]::Min($Fps * $Speed, 4.0)
    $Sampled = Get-Sampled ([int][math]::Floor($Duration / $Speed * $OutFps)) $OutFps
    $Scale = [math]::Min(1.0, [math]::Sqrt($Limit / ([double]$Sampled * $Stream.width * $Stream.height)) * 0.98)
    $Width = [int][math]::Floor($Stream.width * $Scale / 2) * 2
    $Height = [int][math]::Floor($Stream.height * $Scale / 2) * 2
    $Invariant = [Globalization.CultureInfo]::InvariantCulture
    $Filter = "fps=$($OutFps.ToString($Invariant)),scale=${Width}:${Height}"
    if ($Speed -gt 1.0) { $Filter = "setpts=PTS/$($Speed.ToString($Invariant))," + $Filter }
    $Fitted = Join-Path $OutDir ("{0}-fit-{1}x{2}.mp4" -f [IO.Path]::GetFileNameWithoutExtension($Video), $Width, $Height)
    ffmpeg -y -loglevel error -i $Video -vf $Filter -an -c:v libx264 -preset veryfast -crf 20 $Fitted
    if ($LASTEXITCODE -ne 0) { throw "ffmpeg could not prepare $Video" }
    $Note = "${Width}x${Height}"
    if ($Speed -gt 1.0) {
        $Note += (", sped up {0:N2}x to fit the engine's 600 s limit (times the model reports are compressed {0:N2}x)" -f $Speed)
    }
}

Write-Host ("Video:      {0}" -f $Fitted)
Write-Host ("            {0:N0} s source, {1} frames sampled, sent as {2}" -f $Duration, $Sampled, $Note)
if ($Transcript) { Write-Host ("Transcript: {0}" -f $Transcript) }
elseif ($Source -match '^https?://') { Write-Host "Transcript: none (no English subtitles)" }
