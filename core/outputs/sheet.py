"""A viewing sheet: the folder's images in a grid with captions. Usage: sheet.py <out.jpg> <columns> <tile width> <file[:caption]>…"""
import sys
from PIL import Image, ImageDraw, ImageFont

out, cols, tw = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
items = []
for a in sys.argv[4:]:
    p, _, cap = a.partition(":")
    items.append((p, cap or p.rsplit("/", 1)[-1]))
ims = [Image.open(p).convert("RGB") for p, _ in items]
th = int(tw * ims[0].height / ims[0].width)
font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 15)
rows = (len(ims) + cols - 1) // cols
sheet = Image.new("RGB", (cols * tw, rows * (th + 20)), "white")
for i, (im, (_, cap)) in enumerate(zip(ims, items)):
    x, y = (i % cols) * tw, (i // cols) * (th + 20)
    sheet.paste(im.resize((tw, th), Image.LANCZOS), (x, y + 20))
    ImageDraw.Draw(sheet).text((x + 5, y + 2), cap, fill=(0, 0, 0), font=font)
sheet.save(out, quality=88)
print(out, sheet.size)
