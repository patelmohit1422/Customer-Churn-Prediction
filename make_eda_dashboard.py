from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

eda_dir = Path("artifacts/eda")

files = [
    eda_dir / "churn_distribution.png",
    eda_dir / "churn_vs_contract.png",
    eda_dir / "churn_vs_tenure.png",
    eda_dir / "churn_vs_monthly_charges.png",
    eda_dir / "correlation_heatmap.png",
]

missing = [str(f) for f in files if not f.exists()]
if missing:
    raise FileNotFoundError(
        "Missing EDA plots. Run `python main.py` first. Missing: " + ", ".join(missing)
    )

images = [Image.open(f).convert("RGB") for f in files]
images = [ImageOps.contain(img, (900, 600)) for img in images]

cell_w, cell_h = 920, 620
canvas = Image.new("RGB", (cell_w * 2 + 40, cell_h * 3 + 180), "white")
draw = ImageDraw.Draw(canvas)

draw.text((30, 20), "EDA Dashboard — Customer Churn Prediction", fill="black")
draw.text((30, 55), "Churn distribution, contract, tenure, charges, and correlation", fill="black")

positions = [
    (20, 100), (cell_w + 20, 100),
    (20, 100 + cell_h), (cell_w + 20, 100 + cell_h),
]

for img, pos in zip(images[:4], positions):
    canvas.paste(img, pos)

bottom_x = (canvas.width - images[4].width) // 2
canvas.paste(images[4], (bottom_x, 100 + cell_h * 2))

out_path = eda_dir / "eda_dashboard.png"
canvas.save(out_path)
print(f"Saved: {out_path}")