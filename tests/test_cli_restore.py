import tempfile
import unittest
from pathlib import Path
from PIL import Image
import numpy as np
import tifffile
from aether.mobile.cli import main

class TestCLIRestore(unittest.TestCase):
    def test_render_small_image(self):
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp); source=d/'input.png'; output=d/'final.tif'
            Image.fromarray(np.full((12,18,3),[80,120,200],dtype=np.uint8),'RGB').save(source)
            r=main(['restore','--source',str(source),'--output',str(output),
                    '--workdir',str(d/'stage'),'--width','48','--height','32',
                    '--tile','32','--reserve-free-gb','0'])
            self.assertEqual(r,0)
            with tifffile.TiffFile(output) as f:
                self.assertTrue(f.is_bigtiff)
                self.assertEqual(f.pages[0].shape,(32,48,3))
                self.assertEqual(f.pages[0].dtype,np.dtype('uint16'))
if __name__=='__main__':
    unittest.main()
