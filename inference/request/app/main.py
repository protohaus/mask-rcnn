#https://www.machinecurve.com/index.php/2020/03/19/tutorial-how-to-deploy-your-convnet-classifier-with-keras-and-fastapi/ based on this

import uvicorn
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import tensorflow as tf
from typing import List
import io
import numpy as np
import sys

app = FastAPI()

model = tf.keras.models.load_model('./model/SDGModel_2-E100_segmented', compile = True)
model.summary()

# Get the input shape for the model layer
input_shape = model.layers[0].input_shape

# Define the Response
class Prediction(BaseModel):
  filename: str
  contenttype: str
  prediction: List[float] = []
  likely_class: str

class_labels = ['Apple___healthy', 'Blueberry___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 
'Soybean___healthy', 'Squash___Powdery_mildew', 'Tomato___Bacterial_spot', 'Tomato___Late_blight', 
'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
'Tomato___healthy']

# Define the main route
@app.get('/')
def root_route():
  return { 'error': 'Use POST /prediction instead of the root route!' }


def format_image(image: Image, side_length: int) -> Image:
  """Crop to a centered square, resize to the side_length and remove alpha."""

  short_side_length = min(image.size)
  width, height = image.size

  # Create the centered square coordinates
  left = (width - short_side_length)/2
  top = (height - short_side_length)/2
  right = (width + short_side_length)/2
  bottom = (height + short_side_length)/2

  # Crop the center of the image
  image = image.crop((left, top, right, bottom))

  # Resize the image
  image = image.resize((side_length, side_length))

  # Convert from RGBA to RGB *to remove alpha channels*
  if image.mode == 'RGBA':
    image = image.convert('RGB')
  return image

# Define the /prediction route
@app.post('/prediction/', response_model=Prediction)
async def prediction_route(file: UploadFile = File(...)):

  # Ensure that this is an image
  if file.content_type.startswith('image/') is False:
    raise HTTPException(status_code=400, detail=f'File \'{file.filename}\' is not an image.')

  try:
    # Read image contents
    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents))
    pil_image = format_image(pil_image, 256)

    # Convert image into numpy format
    numpy_image = np.array(pil_image)

    # Generate prediction
    prediction_array = np.array([numpy_image])
    predictions = model.predict(prediction_array)
    prediction = predictions[0]
    likely_class = np.argmax(prediction)

    return {
      'filename': file.filename,
      'contenttype': file.content_type,
      'prediction': prediction.tolist(),
      'likely_class': class_labels[likely_class]
    }
  except Exception as err:
    e = sys.exc_info()[1]
    raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8001)