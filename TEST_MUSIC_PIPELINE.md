# Music Pipeline Testing Guide

## Quick Test of Background Music (Skip Face Tracking)

After you've already run the pipeline once to generate cached files, you can now quickly test the background music pipeline without waiting for face tracking.

### Requirements
First, run the pipeline normally at least once to generate cached files:
```bash
python main.py --input video.mp4
```

This will create the following cached files in the `temp/` folder:
- `{video_name}_vertical.mp4` - Vertical video from face tracking
- `{video_name}_transcription.json` - Transcription data

### Test the Music Pipeline

Once you have cached files, you can test music mixing quickly with:

```bash
python main.py --input video.mp4 --skip-face-tracking
```

This will:
1. ✓ Skip all face tracking (saves ~2-5 minutes)
2. ✓ Load the cached vertical video
3. ✓ Skip subtitle/hook generation
4. ✓ Go directly to the music mixing pipeline
5. ✓ Output: `output/{video_name}_final_with_music.mp4`

### What Gets Tested

The skip-face-tracking mode tests:
- Audio extraction from video
- Music file conversion
- Python-based audio mixing (soundfile/pydub)
- Audio normalization
- Remuxing with FFmpeg

### Common Options

Test with custom music volume:
```bash
python main.py --input video.mp4 --skip-face-tracking --music-volume 0.5
```

Test without Discord music selection (use default):
```bash
python main.py --input video.mp4 --skip-face-tracking --no-discord-music
```

Test with audio ducking disabled:
```bash
python main.py --input video.mp4 --skip-face-tracking --no-ducking
```

Test with specific music file:
```bash
python main.py --input video.mp4 --skip-face-tracking --music-file alternative_music.mp3
```

### Troubleshooting

If you get "Cached vertical video not found" error:
- Make sure you've run the full pipeline at least once with that video
- Check that `temp/{video_name}_vertical.mp4` exists

If audio mixing fails:
- Make sure `soundfile` is installed: `pip install soundfile`
- Or `pydub` will be used as fallback
- Check FFmpeg is installed and in PATH

### Output Files

- `output/{video_name}_final.mp4` - Vertical video (no music, no subtitles)
- `output/{video_name}_final_with_music.mp4` - Vertical video WITH background music

## Development Notes

The music pipeline uses:
1. **FFmpeg** - For format conversion and remuxing
2. **soundfile** - For reading/writing WAV files
3. **NumPy** - For simple linear audio mixing
4. **pydub** - Fallback audio mixing if soundfile unavailable

The approach avoids complex FFmpeg filter chains that were causing artifacts.
