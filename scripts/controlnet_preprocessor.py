from controlnet_aux import CannyDetector, OpenposeDetector, MidasDetector, HEDdetector
from controlnet_aux.util import resize_image

class Preprocessor:
    def __init__(self, control_type=None, path="lllyasviel/Annotators"):
        self.control_type = control_type
        if control_type == "canny":
            self.preprocessor = CannyDetector()
        elif control_type == "openpose":
            self.preprocessor = OpenposeDetector.from_pretrained(path)
        elif control_type == "midas":
            self.preprocessor = MidasDetector.from_pretrained(path)
        elif control_type == "hed":
            self.preprocessor = HEDdetector.from_pretrained(path)
        elif control_type == "tile":
            self.preprocessor = None
        else:
            raise NotImplementedError

    def __call__(self, x, params={"t1": 50, "t2": 100, "include_hand": True, "include_face": True}, image_resolution=512):
        if self.control_type == "canny":
            return self.preprocessor(x, 
                params["t1"], 
                params["t2"], 
                image_resolution=image_resolution, 
                output_type="np")

        elif self.control_type == "openpose":
            return self.preprocessor(x, 
                include_hand=params["include_hand"], 
                include_face=params["include_face"], 
                image_resolution=image_resolution, 
                output_type="np")

        elif self.control_type == "tile":
            return resize_image(x, resolution=image_resolution)

        else:
            return self.preprocessor(x, 
                image_resolution=image_resolution, 
                output_type="np")
            
            
