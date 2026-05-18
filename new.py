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

# put the 5th plot centered at the bottom
bottom_x = (canvas.width - images[4].width) // 2
canvas.paste(images[4], (bottom_x, 100 + cell_h * 2))

canvas.save(eda_dir / "eda_dashboard.png")
print("Saved:", eda_dir / "eda_dashboard.png")