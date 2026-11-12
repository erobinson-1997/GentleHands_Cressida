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

Steps (switch on/off in `STEPS` at the top of the script): four mud flaps, bare/rusty bash bar,
California plate with a made-up, mosaic-censored number (`PLATE_TEXT`, `PLATE_MOSAIC`, `FRONT_PLATE`), the crease
ahead of the fuel door, worn mirrors, the TIRE STREETS sticker on the left rear door glass, a faint hood fade (also
written to `skins/event_silver/carpaint_diffuse.dds`), door number panels (`RACE_NUMBER`), the US-style front bumper
without the lip/valance, the panel under it with two amber lamps, the black grille with crest and TOYOTA badge, clear
headlight lenses with amber corners, and the broken outer end of the right tail light (a crack across the red outer sheet, two pieces of it missing so the clear mosaic plastic behind shows, scrape marks), and the paint rubbed through to primer and bare metal on the trunk lid and body corner above it, and the dark double pinstripe along the beltline and across the back of the trunk (`stripe_y()`: straight runs along the doors, the trunk and the front fender, laid out on the panels as they were before the crash (`uncrashed()`) so it follows the bent metal; `trunk_face` first takes the stage-1 crumple back off the rear face of the trunk lid, which is still flat on the real car; `STRIPE_BANDS`). Interior (from `Reference Videos/Interrior.mp4`): `rear_seat_delete` (rear bench out, bare painted metal pan and bulkhead), `bucket_seat` (faded pink bucket seat with harness for the driver; shape in `SEAT_PATH`; harness slots cut through the backrest with rubber grommets (`SLOT_HALF`), rounded edges with the cloth cover wrapped round the sides and over the top (`WRAP_U`), glossy black glitter shell showing in the open back panel, NRG Prisma print on the headrest), `shifter` (manual lever with a green drawstring cloth bag as boot - gathered hem, eyelets, cord, gold script - and the leather TRD knob from the AE86: badge, baseball stitching, shift pattern on the crown), `floor_mat` (checker mat in the driver's footwell), `head_unit` (aftermarket single-DIN head unit with an open pocket below, replacing the stock radio and cassette deck), `passenger_seat` (stock grey velour seat; shape in `PASS_CUSHION`, `PASS_BACK`, `PASS_HEAD`), `interior_colors` (charcoal upper dash and door cards, grey lower dash and door inserts, light grey headliner; colours in `TRIM_CHARCOAL`, `TRIM_GREY`, `HEADLINER_GREY`, `DOOR_CLOTH`).

- kn5io.py     full kn5 load/edit/save (round trip is byte-identical); mesh split, subdivide-safe deform helpers
- preview.py   software render of a kn5 for a quick look without the game: `python scripts/preview.py <kn5> <out_dir> [view ...]`

Front-end grime (`bug_splatter`): dead bugs and a dusty road film on the forward-facing bumper, grille and hood nose (`BUG_SPLATTER`, texture
`gh_bugs.png`; clear in the model, filled in by the pine skin only).

Side mouldings (`side_trim`): the chrome cap on top of the black side moulding (front bumper, fenders, doors, rear quarters) is
black with a thin white accent line, as on the real car. Material `gh_trim_accent` (a copy of `black` with `gh_trim_accent.png`);
the line sits where the cap faces outward (texture v comes from the world normal's height).

### Skins

`build_video_details.py` also writes two skins next to `skins/event_silver` (Event Silver, the base skin: stickers, silver AVS Model 5 wheels):

- `clean_silver` (Clean Silver) - clear `gh_sticker.png` and `gh_number_panel.png` hide the window sticker and the door numbers.
- `mountain_silver` (Mountain Silver) - no stickers either; `gh_debris.png` (pine needles, sap, grit) shows on `DEBRIS_LAYER`, a shell just above the
  top of the car that is clear in the model itself; `gh_bugs.png` (bugs and road film on the front end); flat wheel textures (`MKII_Grey.png`, `MKIII_Grey.png`, `SSR_*_Map.png`) make the AVS wheels matte black.

A skin can only swap textures, so a different wheel design needs its own 3D model (and its own car folder, or CSP); skins only recolour the AVS wheels. `preview.jpg` / `livery.png` are copied from the base skin only when missing:
regenerate the previews in Content Manager.
