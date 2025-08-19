import argparse
import subprocess

def generate_ai_cover(args):
    command = [
        "python3",
        "src/main.py",
        "-i", args.song_input,
        "-dir", args.rvc_dirname,
        "-p", str(args.pitch_change),
        "-pall", str(args.pitch_change_all),
        "-k",
        "-ir", str(args.index_rate),
        "-fr", str(args.filter_radius),
        "-rms", str(args.remix_mix_rate),
        "-palgo", args.pitch_detection_algo,
        "-hop", str(args.crepe_hop_length),
        "-pro", str(args.protect),
        "-mv", str(args.main_vol),
        "-bv", str(args.backup_vol),
        "-iv", str(args.inst_vol),
        "-rsize", str(args.reverb_size),
        "-rwet", str(args.reverb_wetness),
        "-rdry", str(args.reverb_dryness),
        "-rdamp", str(args.reverb_damping),
        "-oformat", args.output_format
    ]

    # Open a subprocess and capture its output
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

    # Print the output in real-time
    for line in process.stdout:
        print(line, end='')

    # Wait for the process to finish
    process.wait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an AI cover song.")
    parser.add_argument("--song_input", type=str, default='https://www.youtube.com/watch?v=ICCgV4ZZEhE', help="Link to a YouTube video or the filepath to a local mp3/wav file to create an AI cover of")
    parser.add_argument("--rvc_dirname", type=str, default='Jungkook', help="Name of the folder in the rvc_models directory containing the RVC model file and optional index file to use")
    parser.add_argument("--pitch_change", type=int, default=0, help="Change the pitch of AI Vocals only. Generally, use 1 for male to female and -1 for vice-versa. (Octaves)")
    parser.add_argument("--pitch_change_all", type=int, default=-4, help="Change the pitch/key of vocals and instrumentals. Changing this slightly reduces sound quality")
    parser.add_argument("--index_rate", type=float, default=0.75, help="A decimal number e.g. 0.5, used to reduce/resolve the timbre leakage problem. If set to 1, more biased towards the timbre quality of the training dataset")
    parser.add_argument("--filter_radius", type=int, default=3, help="A number between 0 and 7. If >=3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness.")
    parser.add_argument("--pitch_detection_algo", choices=["rmvpe", "mangio-crepe"], default="rmvpe", help="Best option is rmvpe (clarity in vocals), then mangio-crepe (smoother vocals).")
    parser.add_argument("--crepe_hop_length", type=int, default=128, help="If pitch detection algo is mangio-crepe, controls how often it checks for pitch changes in milliseconds. The higher the value, the faster the conversion and less risk of voice cracks, but there is less pitch accuracy. Recommended: 128.")
    parser.add_argument("--protect", type=float, default=0.33, help="A decimal number e.g. 0.33. Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy.")
    parser.add_argument("--remix_mix_rate", type=float, default=0.25, help="A decimal number e.g. 0.25. Control how much to use the original vocal's loudness (0) or a fixed loudness (1).")
    parser.add_argument("--main_vol", type=int, default=0, help="Volume change for AI main vocals in decibels. Use -3 to decrease by 3 decibels and 3 to increase by 3 decibels")
    parser.add_argument("--backup_vol", type=int, default=0, help="Volume change for backup vocals in decibels")
    parser.add_argument("--inst_vol", type=int, default=0, help="Volume change for instrumentals in decibels")
    parser.add_argument("--reverb_size", type=float, default=0.15, help="Reverb room size between 0 and 1")
    parser.add_argument("--reverb_wetness", type=float, default=0.2, help="Reverb wet level between 0 and 1")
    parser.add_argument("--reverb_dryness", type=float, default=0.8, help="Reverb dry level between 0 and 1")
    parser.add_argument("--reverb_damping", type=float, default=0.7, help="Reverb damping between 0 and 1")
    parser.add_argument("--output_format", choices=["mp3", "wav"], default="mp3", help="Output format of audio file. mp3 for smaller file size, wav for best quality")
    
    args = parser.parse_args()
    
    generate_ai_cover(args)


