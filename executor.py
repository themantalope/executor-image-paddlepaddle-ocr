print('starting executor.py')

from jina import Executor, DocumentArray, Document, requests
from paddleocr import PaddleOCR
from typing import Optional, Dict, Union
from typing_extensions import Literal
# from jina.logging.predefined import default_logger as logger
import paddleocr
import urllib
import random 
import string
import tempfile
import os 
import io 
import typing
import logging

logger = logging.getLogger(__name__)

# print(f'paddleocr version: {PaddleOCR.__version__}')

print(f'pwd: {os.getcwd()}')

MODES = Literal['ocr', 'struct', 'both']

class PaddlepaddleOCR(Executor):
    """
    An executor to extract text from images using paddlepaddleOCR
    """
    def __init__(
        self,
        paddleocr_args : Optional[Dict] = None,
        copy_uri: bool = True,
        mode: Optional[MODES] = 'ocr',
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
        if isinstance(paddleocr_args, dict):
            self._paddleocr_args.setdefault('use_gpu', paddleocr_args['use_gpu'] if 'use_gpu' in paddleocr_args else True)
        print(f'paddleocr_args: {self._paddleocr_args}')
        super(PaddlepaddleOCR, self).__init__(**kwargs)
        self.model = PaddleOCR(**self._paddleocr_args)
        self.copy_uri = copy_uri
        self.mode = mode
        # print(f'paddleocr version: {PaddleOCR.__version__}')
        # self.logger = logger

    @requests()
    def extract(self, docs: Optional[DocumentArray] = None, **kwargs):
        """
        Load the image from `uri`, extract text and bounding boxes. The text is stored in the  
        `text` attribute of the chunks and the coordinates are stored in the `tags['coordinates']` as a list. 
        The `tags['coordinates']`  contains four lists, each of which corresponds to the `(x, y)` coordinates one corner of the bounding box. 
        :param docs: the input Documents with image URI in the `uri` field
        """
        
        # TODO: allow the user to pass the image as a blob in the request, will filter by the mime type
        # TODO: if the image is a blob, temporarily save it to a file and pass the file name to the model
        missing_doc_ids = []
        if docs is None:
            return
        
        for doc in docs:
            if not doc.uri :
                missing_doc_ids.append(doc.id)
                continue

            with tempfile.TemporaryDirectory() as tmpdir:
                source_fn = (
                    self._save_uri_to_tmp_file(doc.uri, tmpdir)
                    if self._is_datauri(doc.uri)
                    else doc.uri
                )
                for r in self.model.ocr(source_fn, cls=True):
                    # logger.info(f'paddle model result: {r}')
                    # logger.info(f'paddle model result type: {type(r)}')
                    # print('paddle model result: ', r)
                    # print(r)
                    # print(type(r))
                    # print(r[0])
                    for dets in r:
                        coord, (text, score) = dets
                        c = Document(text=text, weight=score)
                        c.tags['coordinates'] = coord
                        if self.copy_uri:
                            c.tags['img_uri'] = doc.uri
                        doc.chunks.append(c)
        if missing_doc_ids  :
            logger.warning(f'No uri passed for the following Documents:{", ".join(missing_doc_ids)}')

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