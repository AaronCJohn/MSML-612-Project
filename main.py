"""
Pokémon Diffusion, a unified entry point.

Usage:
    
    Baseline Inference:
        python main.py inference baseline --style sprite
        python main.py inference baseline --style 3d

    Conditional Inference:
        python main.py inference conditional --type water --style sprite
        python main.py inference conditional --type grass poison --style 3d
        python main.py inference conditional --type fire --style sprite --stage "evo 1"

    Train:
        python main.py train --arch conditional
        python main.py train --arch baseline

Defaults (no arguments): inference conditional --type water --style sprite --stage base
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Pokémon Diffusion: train or generate images",
    )
    top = parser.add_subparsers(dest="command")

    # Inference
    infer_p = top.add_parser("inference", help="Generate Pokémon images")
    infer_sub = infer_p.add_subparsers(dest="arch")

    # Inference Baseline
    base_p = infer_sub.add_parser("baseline", help="Unconditional baseline model")
    base_p.add_argument(
        "--style", default="sprite",
        choices=["3d", "sprite"],
        help="Art style, selects the baseline checkpoint  (default: sprite)",
    )

    # Inference Conditional
    cond_p = infer_sub.add_parser("conditional", help="Conditional model")
    cond_p.add_argument(
        "--type", nargs="+", default=["water"],
        help="Pokémon type(s)  (default: water)",
    )
    cond_p.add_argument(
        "--style", default="sprite",
        choices=["3d", "sugimori", "sprite"],
        help="Art style  (default: sprite)",
    )
    cond_p.add_argument(
        "--stage", default="base",
        choices=["base", "evo 1", "evo 2"],
        help="Evolution stage  (default: base)",
    )

    # Train
    train_p = top.add_parser("train", help="Train a diffusion model")
    train_p.add_argument(
        "--arch", default="conditional",
        choices=["baseline", "conditional"],
        help="Architecture  (default: conditional)",
    )

    # Defaults (no arguments)
    if len(sys.argv) == 1:
        args = parser.parse_args(["inference", "conditional"])
    else:
        args = parser.parse_args()

    if args.command == "train":
        from model.train import main as train_main
        train_main(arch=args.arch)
    elif args.command == "inference":
        from model.inference import main as inference_main
        arch = args.arch or "conditional"
        gen_types = getattr(args, "type", ["water"])
        gen_style = getattr(args, "style", "sprite")
        gen_stage = getattr(args, "stage", "base")
        inference_main(arch=arch, gen_types=gen_types, gen_style=gen_style, gen_stage=gen_stage)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
