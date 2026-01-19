from PIL import Image


class ImageProcessor:
    
    @classmethod
    def apply_gray_mask(cls, img, mask) -> dict:
        if mask is not None:
            img = img.convert("RGB")
            mask = mask.convert("L")
            gray_color = (128, 128, 128) 
            gray_img = Image.new("RGB", img.size, gray_color)
            img = Image.composite(gray_img, img, mask)