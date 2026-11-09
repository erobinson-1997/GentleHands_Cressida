"""Paths used by every script. Edit here, or override with environment variables.

Expected project layout (the folder that contains this scripts/ folder):

  GentleHands_Cressida/
    scripts/                      <- these files
    originals/                    <- a PRISTINE copy of the original mod's  ggh_toyota_cressida_mx83_stage_1.kn5
    reference/                    <- advan_model_5s.fbx  ("Advan AVS 5" by gunis.david, CC BY 4.0)
    dist/                         <- build output goes here (nothing is overwritten in the car folder)
    gentlehands_toyota_cressida_mx83/   <- the car folder you copy the result into
"""
import os
from pathlib import Path
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
ORIGINAL_KN5=Path(os.environ.get('CRESSIDA_ORIGINAL_KN5',PROJECT/'originals'/'ggh_toyota_cressida_mx83_stage_1.kn5'))
REF_FBX=Path(os.environ.get('CRESSIDA_REF_FBX',PROJECT/'reference'/'advan_model_5s.fbx'))
OUTPUT_KN5=Path(os.environ.get('CRESSIDA_OUTPUT_KN5',PROJECT/'dist'/'ggh_toyota_cressida_mx83_stage_1.kn5'))
