# Build scripts

These rebuild the modified `ggh_toyota_cressida_mx83_stage_1.kn5` from the ORIGINAL author's model. They edit the binary
KN5 directly (no Blender needed). Nothing in your car folder is overwritten: output goes to `dist/`.

## Layout (project root = the folder that contains `scripts/`)

    GentleHands_Cressida/
      scripts/        these files
      originals/      a PRISTINE copy of the original mod's ggh_toyota_cressida_mx83_stage_1.kn5   <- you must add this
      reference/      advan_model_5s.fbx  ("Advan AVS 5" by gunis.david, CC BY 4.0)                <- you must add this
      dist/           created by the build; contains the finished .kn5
      gentlehands_toyota_cressida_mx83/   the car folder (copy dist/*.kn5 into it)

The pristine original is the file BEFORE any of our edits. Your repo's git history has it
(`git show <first-commit>:path/to/ggh_toyota_cressida_mx83_stage_1.kn5 > originals/ggh_toyota_cressida_mx83_stage_1.kn5`).
Paths can be changed in `config.py` or with the environment variables CRESSIDA_ORIGINAL_KN5, CRESSIDA_REF_FBX, CRESSIDA_OUTPUT_KN5.

## Run

    pip install -r scripts/requirements.txt
    python scripts/build_rear_damage.py            (use `py` on Windows if `python` isn't found)

Then copy `dist/ggh_toyota_cressida_mx83_stage_1.kn5` over the one in the car folder (keep a backup first).
The build is deterministic: the same inputs give a byte-identical file (MD5 e02e7a7c68806a991ada7cf014fc2f8e for the current version, built with Python 3.15, numpy 2.5.4, scipy 1.18.1, Pillow 13.0.0; other library versions may differ by float rounding).

## Modules
- build_rear_damage.py  main build: crash deformation, bumper -> rear panel, bash bar, trunk liner, exhaust, tow-hook removal, embeds the silver paint, runs the steps below
- wheels.py            wheels from the reference FBX (scale, orientation, smooth normals, UV patches, nuts, cap, spacers); needs reference/advan_model_5s.fbx
- rotor.py             brake rotors (replace the small hub piece; uses the car's own `disc` material)
- paint.py             silver tint for carpaint_diffuse.dds (mipmaps built from colour only; original DDS header kept)
- casehard.py          procedural case-hardened exhaust texture
- kn5.py, geo.py, kn5build.py   KN5 reader, original-model geometry helpers, mesh/node writers
- fbxread.py, fbxmesh.py        minimal binary FBX reader

## Credit (CC BY 4.0)
"Advan AVS 5" (https://skfb.ly/oE7Rn) by gunis.david, licensed CC BY 4.0 (http://creativecommons.org/licenses/by/4.0/).
Changes: converted to KN5, scaled and re-proportioned to the car, re-oriented, re-shaded and re-textured; lug nuts, centre cap and spacers added.

## Stage 2: details from the walk-around videos

    python scripts/build_video_details.py

Takes the stage-1 model (the output of `build_rear_damage.py`, MD5 e02e7a7c...) from `originals/stage1_rear_damage.kn5`
and writes `dist/ggh_toyota_cressida_mx83_stage_1.kn5`. The stage-1 file is the kn5 committed at cc7ebdd, so it can be
restored with `git show cc7ebdd:gentlehands_toyota_cressida_mx83/ggh_toyota_cressida_mx83_stage_1.kn5 > originals/stage1_rear_damage.kn5`
(run that from Git Bash or cmd, not PowerShell, which corrupts binary redirects). No FBX is needed for this stage.

Steps (switch on/off in `STEPS` at the top of the script): satin-black wheels, four mud flaps, bare/rusty bash bar,
California plate with a made-up, mosaic-censored number (`PLATE_TEXT`, `PLATE_MOSAIC`, `FRONT_PLATE`), the crease
ahead of the fuel door, worn mirrors, the TIRE STREETS sticker on the left rear door glass, a faint hood fade (also
written to `skins/silver_176/carpaint_diffuse.dds`), door number panels (`RACE_NUMBER`), the US-style front bumper
without the lip/valance, the panel under it with two amber lamps, the black grille with crest and TOYOTA badge, clear
headlight lenses with amber corners, and the broken outer end of the right tail light (a crack across the red outer sheet, two pieces of it missing so the clear mosaic plastic behind shows, scrape marks), and the paint rubbed through to primer and bare metal on the trunk lid and body corner above it, and the dark double pinstripe along the beltline and across the back of the trunk (`STRIPE_Y`, `STRIPE_BANDS`).

- kn5io.py     full kn5 load/edit/save (round trip is byte-identical); mesh split, subdivide-safe deform helpers
- preview.py   software render of a kn5 for a quick look without the game: `python scripts/preview.py <kn5> <out_dir> [view ...]`
