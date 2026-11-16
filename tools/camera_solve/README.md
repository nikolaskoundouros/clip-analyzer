# camera_solve: scripts from the 14-15 Nov 2026 run 021 session (blender-funnel), saved from a temporary folder

They ran as written on that day; paths inside them still point at that session's folders and need
adjusting before reuse. Measured results are in ~/Projects/blender-funnel/runs/021-reference-shots/review.md.

- solve_shot_free_lens.py: COLMAP (pycolmap 4.2.1, BSD) on one shot's frames; lens solved. Box venv /Users/nick/venvs/colmap.
- solve_all_shots.py: COLMAP per shot with the lens fixed at 961.4 px (540x960); writes <shot>/path.json with quality numbers.
- solve_masked_people.py: same with people masked (face boxes + hair). FAILED on 1.9-4.1 s (blur, 55% masked).
- orbit_round_subject.py / _ours.py: camera angle round the subject (subject = solved points on the orange hair).
- compare_render_vs_reference.py, compare_shot_three_way.py, compare_0_10s.py: our render vs the reference
  (side-by-side or separate videos, frame sheets, clip-analyzer readings on both, scores).
- score_move_versions.py: render several versions of one move, analyse each like the reference, keep the closest.
