#!/bin/bash

# Test script for separated audio files in tmp/ folder
# Usage: ./test_tmp_files.sh [voice_model_name] [pitch_change_all]

# Default values
VOICE_MODEL=${1:-"Jimin"}
PITCH_CHANGE_ALL=${2:-0}

echo "🎵 Testing AI Cover Generation with tmp/ files"
echo "📁 Voice Model: $VOICE_MODEL"
echo "🎹 Pitch Change All: $PITCH_CHANGE_ALL"
echo "----------------------------------------"

# Check if files exist
if [ ! -f "tmp/Vocals_No_Noise.wav" ]; then
    echo "❌ Error: tmp/Vocals_No_Noise.wav not found"
    exit 1
fi

if [ ! -f "tmp/Instrumental.wav" ]; then
    echo "❌ Error: tmp/Instrumental.wav not found"
    exit 1
fi

# Check if voice model exists
if [ ! -d "rvc_models/$VOICE_MODEL" ]; then
    echo "❌ Error: Voice model directory 'rvc_models/$VOICE_MODEL' not found"
    echo "Available models:"
    ls -1 rvc_models/ | grep -v "hubert_base.pt\|rmvpe.pt\|MODELS.txt\|public_models.json"
    exit 1
fi

echo "✅ Files and model verified"
echo "🚀 Starting AI cover generation..."
echo ""

# Run the test
python3 test_separated_audio.py \
    --voice_input "tmp/Vocals_No_Noise.wav" \
    --inst_input "tmp/Instrumental.wav" \
    --rvc_dirname "$VOICE_MODEL" \
    --pitch_change_all "$PITCH_CHANGE_ALL" \
    --output_format "mp3"

echo ""
echo "🎉 Test completed!"
