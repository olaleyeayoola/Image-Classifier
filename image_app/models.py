from django.db import models
from PIL import Image
# Create your models here.
class Search(models.Model):
    name = models.CharField(max_length = 255)
    image = models.ImageField(upload_to = 'searchimage')

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)

        img = Image.open(self.image.path)

        if img.height > 256 or img.weight > 256:
            output_size = (256, 256)
            img.thumbnail(output_size)
            img.save(self.image.path)

    def get_image_name(self):
        return self.image.name