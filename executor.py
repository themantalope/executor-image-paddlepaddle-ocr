print('starting executor.py')

from jina import Executor, DocumentArray, Document, requests
from paddleocr import PaddleOCR
from typing import Optional, Dict, Union
from typing_extensions import Literal
# from jina.logging.predefined import default_logger as logger
import paddleocr
from paddleocr import PaddleOCR, draw_ocr, PPStructure, draw_structure_result, save_structure_res
import uuid
import urllib
import random 
import string
import tempfile
import os 
import io 
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# print(f'paddleocr version: {PaddleOCR.__version__}')

FONT = 'helvetica-light.ttf'
assert os.path.isfile(FONT), f'font {FONT} not found'

OPT_DICT = {
    'det_model_dir': os.environ['DET_INFER_MODEL_DIR'],
    'rec_model_dir': os.environ['REC_INFER_MODEL_DIR'],
    'cls_model_dir': os.environ['CLS_INFER_MODEL_DIR'],
    'use_dilation': True,
}

print(f'pwd: {os.getcwd()}')

MODES = Literal['ocr', 'struct', 'both']

class PaddlePaddleOCR(Executor):
    """
    An executor to extract text from images using paddlepaddleOCR
    """
    def __init__(
        self,
        paddleocr_args : Optional[Dict] = None,
        copy_uri: bool = True,
        mode: Optional[MODES] = 'ocr',
        save_ocr_images: bool = False,
        **kwargs
        ):
        """
        :param paddleocr_args: the arguments for `paddleocr` for extracting text. By default
        `use_angle_cls=True`,
        `lang='en'` means the language you want to extract, 
        `use_gpu=False` whether you want to use gpu or not.
        Other params can be found in `paddleocr --help`. More information can be found here https://github.com/PaddlePaddle/PaddleOCR
        :param copy_uri: Set to `True` to store the image `uri` at the `.tags['image_uri']` of the chunks that are extracted from the image.
        By default, `copy_uri=True`. Set this to `False` when you don't want to store image `uri` with the chunks or when the image uri is a `datauri`.
        :param mode: the mode of the executor. `ocr` for extracting text from images and `struct` for doing structure extraction. use 'both' to do both.
            default is 'ocr'
        """
        self._paddleocr_args = paddleocr_args or {}
        self._paddleocr_args.setdefault('use_angle_cls', True) 
        self._paddleocr_args.setdefault('lang', 'en')
        self._paddleocr_args.update(OPT_DICT)
        if isinstance(paddleocr_args, dict):
            self._paddleocr_args.setdefault('use_gpu', paddleocr_args['use_gpu'] if 'use_gpu' in paddleocr_args else True)
        print(f'paddleocr_args: {self._paddleocr_args}')
        super(PaddlePaddleOCR, self).__init__(**kwargs)
        self.model = PaddleOCR(**self._paddleocr_args)
        self.table_engine = PPStructure(show_log=True, image_orientation=True)
        self.copy_uri = copy_uri
        self.mode = mode
        self.logger = logger
        self.save_ocr_images = save_ocr_images
        # print(f'paddleocr version: {PaddleOCR.__version__}')
        # self.logger = logger

    def _save_doc_image_tensor_to_temp_file(self, doc, tmpdir, ext='png'):
        assert ext in ['png', 'jpeg'], f'extension {ext} not supported. must be png or jpeg'
        tmp_fn = os.path.join(
            tmpdir,
            f'{str(uuid.uuid4())}.{ext}'
        )
        doc.save_image_tensor_to_file(tmp_fn, image_format=ext)
        return tmp_fn
        
    
    def _convert_ocr_results_to_dict(self, ocr_results):
        ocr_dicts = []
        for dets in ocr_results:
            d = {}
            coords, (text, score) = dets
            d['text'] = text
            d['score'] = score
            d['coords'] = coords
            d['upper_left'] = coords[0]
            d['upper_right'] = coords[1]
            d['lower_right'] = coords[2]
            d['lower_left'] = coords[3]
            d['width'] = max(d['upper_right'][0], d['lower_right'][0]) - min(d['upper_left'][0], d['lower_left'][0])
            d['height'] = max(d['lower_left'][1], d['lower_right'][1]) - min(d['upper_left'][1], d['upper_right'][1])
            d['center'] = np.array(coords).mean(axis=0)
            ocr_dicts.append(d)
        return ocr_dicts
    
    def _convert_structure_results_to_dict(self, structure_results):
        pass # TODO: implement this
            
    
    @requests()
    def extract(self, docs: Optional[DocumentArray] = None, **kwargs):
        """
        saves the image tensor of the documents to a temporary file and then extracts text from the image using paddlepaddleOCR
        
        """
        
        # TODO: allow the user to pass the image as a tensor in the request, will filter by the mime type
        # missing_doc_ids = []
        missing_tensor_doc_ids = []
        if docs is None:
            return
        
        for doc in docs:
            # if not doc.uri :
            #     missing_doc_ids.append(doc.id)
            #     continue
            if doc.tensor is None:
                missing_tensor_doc_ids.append(doc.id)
                

            with tempfile.TemporaryDirectory() as tmpdir:
                # source_fn = (
                #     self._save_uri_to_tmp_file(doc.uri, tmpdir)
                #     if self._is_datauri(doc.uri)
                #     else doc.uri
                # )
                source_fn = self._save_doc_image_tensor_to_temp_file(doc, tmpdir)
                ocr_r = None
                str_r = None
                
                if self.mode == 'ocr':
                    ocr_r = self._get_ocr(source_fn)
                elif self.mode == 'struct':
                    str_r = self._get_structure(source_fn)
                elif self.mode == 'both':
                    ocr_r = self._get_ocr(source_fn)
                    str_r = self._get_structure(source_fn)
                
                if ocr_r and str_r is None:
                    ocr_dicts = self._convert_ocr_results_to_dict(ocr_r)
                    for d in ocr_dicts:
                        c = Document(text=d['text'], weight=d['score'])
                        # c.tags['coordinates'] = d['coords']
                        c.tags.update(d)
                        if self.copy_uri:
                            c.tags['img_uri'] = doc.uri
                        doc.chunks.append(c)
                    if self.save_ocr_images:
                        ocr_vis = self._get_ocr_visualization(source_fn, ocr_r)
                        doc.tags['ocr_image'] = ocr_vis
                elif str_r and ocr_r is None:
                    raise NotImplementedError('structure extraction not implemented yet')
                elif ocr_r and str_r:
                    raise NotImplementedError('structure extraction not implemented yet')
                
                # for r in self.model.ocr(source_fn, cls=True):
                #     logger.info(f'paddle model result: {r}')
                #     logger.info(f'paddle model result type: {type(r)}')
                #     # print('paddle model result: ', r)
                #     # print(r)
                #     # print(type(r))
                #     # print(r[0])
                #     for dets in r:
                #         coord, (text, score) = dets
                #         c = Document(text=text, weight=score)
                #         c.tags['coordinates'] = coord
                #         if self.copy_uri:
                #             c.tags['img_uri'] = doc.uri
                #         doc.chunks.append(c)
        
        if missing_tensor_doc_ids:
            logger.warning(f'No uri passed for the following Documents:{", ".join(missing_tensor_doc_ids)}')

    def _get_ocr(self, filepath):
        results = self.model.ocr(filepath, cls=True)
        return results
    
    def _get_ocr_visualization(self, filepath, ocr_results):
        image = Image.open(filepath).convert('RGB')
        boxes = [line[0] for line in ocr_results[0]]
        txts = [line[1][0] for line in ocr_results[0]]
        scores = [line[1][1] for line in ocr_results[0]]
        im_show = draw_ocr(image, boxes, txts, scores, font_path=FONT)
        return im_show
    
    def _get_structure(self, filepath):
        results = self.table_engine(filepath)
        return results
    
    def _get_strcuture_visualization(self, filepath, str_results):
        pass
    
    def _is_datauri(self, uri):
        scheme = urllib.parse.urlparse(uri).scheme
        return scheme in {'data'}

    def _save_uri_to_tmp_file(self, uri, tmpdir):
        req = urllib.request.Request(uri, headers={'User-Agent': 'Mozilla/5.0'})
        tmp_fn = os.path.join(
            tmpdir,
            ''.join([random.choice(string.ascii_lowercase) for i in range(10)])
            + '.png',
        )
        with urllib.request.urlopen(req) as fp:
            buffer = fp.read()
            binary_fn = io.BytesIO(buffer)
            with open(tmp_fn, 'wb') as f:
                f.write(binary_fn.read())
        return tmp_fn
    
# from paddleocr examples:
# from paddleocr import PaddleOCR,draw_ocr
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.
# ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
# img_path = './imgs_en/img_12.jpg'
# result = ocr.ocr(img_path, cls=True)
# for idx in range(len(result)):
#     res = result[idx]
#     for line in res:
#         print(line)


# # draw result
# from PIL import Image
# result = result[0]
# image = Image.open(img_path).convert('RGB')
# boxes = [line[0] for line in result]
# txts = [line[1][0] for line in result]
# scores = [line[1][1] for line in result]
# im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/simfang.ttf')
# im_show = Image.fromarray(im_show)
# im_show.save('result.jpg')

# import os
# import cv2
# from paddleocr import PPStructure,draw_structure_result,save_structure_res

# table_engine = PPStructure(show_log=True, image_orientation=True)

# save_folder = './output'
# img_path = 'ppstructure/docs/table/1.png'
# img = cv2.imread(img_path)
# result = table_engine(img)
# save_structure_res(result, save_folder,os.path.basename(img_path).split('.')[0])

# for line in result:
#     line.pop('img')
#     print(line)

# from PIL import Image

# font_path = 'doc/fonts/simfang.ttf' # PaddleOCR下提供字体包
# image = Image.open(img_path).convert('RGB')
# im_show = draw_structure_result(image, result,font_path=font_path)
# im_show = Image.fromarray(im_show)
# im_show.save('result.jpg')