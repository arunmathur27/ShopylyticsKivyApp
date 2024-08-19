from kivymd.app import MDApp
from kivy.lang import Builder
from kivymd.uix.screen import MDScreen
from kivy.core.window import Window
from kivy.uix.image import Image
from kivymd.uix.list import MDList,OneLineAvatarListItem,ImageLeftWidget
import time
import numpy
from PIL import Image as Img
from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor



Window.size = (640,640)

Builder.load_file("homescreen.kv")
class Homescreen(MDScreen):
	def __init__(self,**kwargs):
		super().__init__(**kwargs)

		# GET SELECTOR FROM KV FILE CAMERA 
		self.mycamera = self.ids.camera
		self.myimage = Image()
		self.resultbox = self.ids.resultbox
		self.mybox = self.ids.mybox



	def captureyouface(self):
		# CREATE TIMESTAMP NOT FOR YOU FILE IMAGE
		# THIS SCRIPT GET TIME MINUTES AND DAY NOW
		timenow = time.strftime("%Y%m%d_%H%M%S")
		model = YOLO('Demo_Model_Beverage_Category.pt')

		texture = self.mycamera.texture
		size = texture.size
		pixels = texture.pixels
		pil_image = Img.frombytes(mode='RGBA', size=size, data=pixels)
		res = model.predict(pil_image, task="detect", conf=0.60, save_crop=False, show_conf=True, augment=False)[0]
		res = res.plot(font_size=30, line_width=1, conf=True)
		res = res[:, :, ::-1]
		res = Img.fromarray(res)
		justfilename = f"myimage_{timenow}.png"
		res.save(f'.\Output\{justfilename}', quality=99)


		# AND EXPORT YOU CAMERA CAPTURE TO PNG IMAGE
		self.mycamera.export_to_png("myimage_{}.png".format(timenow))
		self.myimage.source = "myimage_{}.png".format(timenow)




		self.resultbox.add_widget(
			OneLineAvatarListItem(
				ImageLeftWidget(
					source="myimage_{}.png".format(timenow),
					size_hint_x=0.3,
					size_hint_y=1,

					# AND SET YOU WIDHT AND HEIGT YOU PHOTO
					size=(640,640)

					),
				text=self.ids.name.text
				)

			)


class MyApp(MDApp):
	def build(self):
		return Homescreen()

if __name__ == "__main__":
	MyApp().run()

