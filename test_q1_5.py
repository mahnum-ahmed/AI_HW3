import os
import Q1_quantize as q

out = "out/q1"
os.makedirs(out, exist_ok=True)
for i in range(1, 6):
    img = q.load_image(f"images/sample{i}.jpg")
    q.part_e_dbscan(img, out, f"sample{i}", min_pts=q.MIN_PTS)