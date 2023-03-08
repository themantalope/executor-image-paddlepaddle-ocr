from paddleocr import PPStructure, PaddleOCR

model = PaddleOCR( use_angle_cls=True, det=True, rec=True, cls=True, use_gpu=True)
table_engine = PPStructure(use_gpu=True, image_orientation=True)

model = PaddleOCR( use_angle_cls=True, det=True, rec=True, cls=True, use_gpu=True, lang='en')
table_engine = PPStructure(use_gpu=True, image_orientation=True, lang='en')