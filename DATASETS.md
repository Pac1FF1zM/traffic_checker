# Dataset shortlist for the WIUT fixed-CCTV task

Checked on 2026-09-23. Always re-read the dataset card and terms before downloading,
training, redistributing, or publishing derived annotations.

## Recommended order

| Priority | Dataset | Best use in this task | Viewpoint / labels | Size and access | Licence / caveat |
|---|---|---|---|---|---|
| 1 | [ACCIDENT](https://accidentbench.github.io/) | `accident` temporal localization and CCTV domain adaptation | Fixed distant surveillance; accident time, location and five collision types | 2,027 real + 2,211 CARLA clips; [Kaggle](https://www.kaggle.com/datasets/picekl/accident) | Check the Kaggle terms before use; this is the closest visual domain to WIUT |
| 2 | [CADP](https://ankitshah009.github.io/accident_forecasting_traffic_camera) | `accident`, early forecasting, small/distant road users | CCTV-like internet videos; 205 clips have full spatio-temporal annotations | 1,416 clips; official page links frames and JSON | Non-commercial research only; other use needs author permission |
| 3 | [AI City 2021 Track 4](https://www.aicitychallenge.org/2021-track4-download/) | Generic traffic-anomaly temporal detection from fixed cameras | Iowa DOT and urban traffic cameras; anomaly start time | More than 50 hours in the associated challenge paper; direct Drive download | Review AI City dataset agreement; strong domain match, broader anomaly labels |
| 4 | [DoTA](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly) | Pretraining accident/anomaly temporal encoder; anomaly type and object | Primarily dashcam; start/end, anomaly class/object, tracks | 4,677 clips, about 55 GB | Repository is MIT; verify source-video terms separately |
| 5 | [Nexar Collision Prediction](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction) | Part B risk calibration, `accident` / `near_miss` anticipation | Dashcam; event time and human alert time | 1,500 x ~40 s, 31.4 GB; gated HF access | Nexar Open Data License; collision and near-collision share one positive label |
| 6 | [CCD](https://github.com/Cogito2012/CarCrashDataset) | Part B and accident clip classifier | Dashcam; per-frame crash labels and environment attributes | 1,500 crashes + 3,000 normal clips | MIT repository; confirm the video redistribution terms |
| 7 | [WTS](https://github.com/woven-visionai/wts-dataset) | `jaywalking`, `failure_to_yield`, pedestrian-vehicle interaction and staged accidents | Ego + fixed overhead views; phases, descriptions, boxes and gaze | 1.2k events / 130+ scenarios; request form | Academic research only; no redistribution or commercial use |
| 8 | [PIE](https://www.data.nvision2.eecs.yorku.ca/PIE_dataset/) | Pedestrian crossing intention and risk features | Dashcam; 1,842 pedestrian tracks, crossings, lights, signs, road boundaries | 6+ hours / 909k frames | Videos and annotations are MIT according to the official page |
| 9 | [UA-DETRAC](https://arxiv.org/abs/1511.04136) | Fine-tune detector/tracker for high-angle fixed traffic CCTV | Fixed cameras; vehicle boxes, tracks, weather and occlusion | 100 sequences / 140k+ frames | No event labels; check the current download agreement |
| 10 | [BDD100K](https://www.bdd100k.com/) | Lane, drivable-area, traffic-light and object pretraining | Dashcam; detection, tracking, lanes, drivable area, traffic-light colour | 100k x 40 s videos / 1,000+ hours | Large domain gap, but excellent auxiliary perception labels; review access terms |

## Secondary options

- [DADA-2000](https://github.com/JWFangit/LOTVS-DADA): about 2,000 accident
  scenarios with attention maps and fine accident categories. Useful for a clip
  encoder, but dashcam domain and a large 53-116 GB download.
- [A3D](https://github.com/MoonBlvd/tad-IROS2019): 1,500 dashcam anomaly clips
  with human temporal annotations. DoTA is its larger successor, so prefer DoTA
  unless reproducing an A3D baseline.
- [JAAD](http://data.nvision2.eecs.yorku.ca/JAAD_dataset/): pedestrian crossing
  behaviour, context and bounding boxes; useful if PIE is too large.

## Mapping to WIUT labels

- `accident`: ACCIDENT, CADP, DoTA, Nexar, CCD, WTS, DADA-2000, A3D.
- `near_miss`: Nexar is the cleanest starting point; DoTA and WTS can add context.
- `jaywalking` / `failure_to_yield`: WTS, PIE and JAAD.
- `wrong_way`, `stopped_vehicle`, `congestion`: AI City anomaly data plus
  self-labelled WIUT samples; these are mostly trajectory rules, not clip classes.
- `red_light`, `stop_line`, `solid_line_crossing`, illegal turns: BDD100K helps
  perception, but the final labels should be generated from WIUT camera geometry
  and manually verified local annotations.
- `road_obstacle`, `fire_smoke`: none of the shortlisted datasets is a clean match;
  curate small public subsets or use a separately licensed open-vocabulary dataset.

## Suggested minimal download plan

1. Start with ACCIDENT and CADP for fixed-camera accident detection.
2. Add Nexar or CCD only for the causal Part B head.
3. Use WTS or PIE only if pedestrian-related labels appear in the WIUT samples.
4. Do not download BDD100K in full initially; use only the relevant detection,
   lane and traffic-light subsets or pretrained perception weights.
